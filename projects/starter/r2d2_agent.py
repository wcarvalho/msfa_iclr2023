# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recurrent DQN (R2D2) agent implementation."""

import copy
from typing import Optional

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import trfl

from acme.agents.tf import actors
from acme.agents.tf.r2d2 import learning
from acme.jax import savers as jax_savers
from acme.jax import utils as jax_utils
import tensorflow as tf

from rljax.projects.starts.r2d2_learner import R2D2Learner

class R2D2(agent.Agent):
  """R2D2 Agent.
  This implements a single-process R2D2 agent. This is a Q-learning algorithm
  that generates data via a (epislon-greedy) behavior policy, inserts
  trajectories into a replay buffer, and periodically updates the policy (and
  as a result the behavior) by sampling from this buffer.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: hk.RNNCore,
      burn_in_length: int,
      trace_length: int,
      replay_period: int,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      discount: float = 0.99,
      batch_size: int = 32,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      epsilon: float = 0.01,
      max_gradient_norm: float = 40,  # For gradient clipping.
      learning_rate: float = 1e-3,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: float = 32.0,
      store_lstm_state: bool = True,
      max_priority_weight: float = 0.9,
      checkpoint: bool = True,
      seed: int = 1,
  ):

    if store_lstm_state:
      import ipdb; ipdb.set_trace()
      extra_spec = {
          'core_state': jax_utils.squeeze_batch_dim(network.initial_state(1)),
      }
    else:
      extra_spec = ()

    sequence_length = burn_in_length + trace_length + 1
    reverb_replay = replay.make_reverb_prioritized_sequence_replay(
      environment_spec=environment_spec,
      n_step=trace_length,
      batch_size=batch_size,
      max_replay_size=max_replay_size,
      min_replay_size=min_replay_size,
      priority_exponent=priority_exponent,
      discount=discount,
      extra_spec=extra_spec,
      burn_in_length=burn_in_length,
      sequence_length=sequence_length,
    )
    self._server = reverb_replay.server

    optimizer = optax.chain(
      optax.clip_by_global_norm(max_gradient_norm),
      optax.adam(learning_rate),
    )

    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(seed))

    learner = R2D2Learner(
        network=network,
        loss_fn=loss_fn,
        burn_in_length=burn_in_length,
        sequence_length=sequence_length,
        data_iterator=reverb_replay.data_iterator,
        optimizer=optimizer,
        target_update_period=config.target_update_period,
        random_key=key_learner,
        replay_client=reverb_replay.client,
    )

    # ======================================================
    # Copied
    # ======================================================
    target_network = copy.deepcopy(network)
    jax_utils.create_variables(network, [environment_spec.observations])
    jax_utils.create_variables(target_network, [environment_spec.observations])

    learner = learning.R2D2Learner(
        environment_spec=environment_spec,
        network=network,
        # target_network=target_network,
        burn_in_length=burn_in_length,
        sequence_length=sequence_length,
        dataset=dataset,
        reverb_client=reverb.TFClient(address),
        counter=counter,
        logger=logger,
        discount=discount,
        target_update_period=target_update_period,
        importance_sampling_exponent=importance_sampling_exponent,
        max_replay_size=max_replay_size,
        learning_rate=learning_rate,
        store_lstm_state=store_lstm_state,
        max_priority_weight=max_priority_weight,
    )

    self._checkpointer = jax_savers.Checkpointer(
        subdirectory='r2d2_learner',
        time_delta_minutes=60,
        objects_to_save=learner.state,
        enable_checkpointing=checkpoint,
    )
    self._snapshotter = jax_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

    policy_network = snt.DeepRNN([
        network,
        lambda qs: trfl.epsilon_greedy(qs, epsilon=epsilon).sample(),
    ])

    actor = actors.RecurrentActor(
        policy_network, adder, store_recurrent_state=store_lstm_state)
    observations_per_step = (
        float(replay_period * batch_size) / samples_per_insert)
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=replay_period * max(batch_size, min_replay_size),
        observations_per_step=observations_per_step)

  def update(self):
    super().update()
    self._snapshotter.save()
    self._checkpointer.save()