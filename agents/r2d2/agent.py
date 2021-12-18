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

"""DQN agent implementation."""

from typing import Tuple

from acme import specs
from acme.agents import agent
from acme.agents import replay
from acme.agents.jax import actors
from acme.agents.jax.dqn import learning_lib
from acme.wrappers import observation_action_reward
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.jax import utils as jax_utils

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax

from agents.r2d2 import losses
# from agents.r2d2 import config
from agents.r2d2 import actor_core as actor_core_lib
# from agents.r2d2 import learning
from agents.r2d2.networks import R2D2Network


class R2D2(agent.Agent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: R2D2Network,
      burn_in_length: int = 40,
      trace_length: int = 80,
      replay_period: int = 40,
      batch_size: int = 16,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 0.5,
      min_replay_size: int = 1000,
      max_replay_size: int = 100000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      max_gradient_norm: float = 40, # TODO: doublecheck
      epsilon: float = 0.05,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      seed: int = 1,
      store_lstm_state: bool = False,
      max_priority_weight: float = 0.9,
  ):
    """Initialize the agent."""
    # Data is communicated via reverb replay.

    if store_lstm_state:
      import ipdb; ipdb.set_trace()
      extra_spec = {
          'core_state': jax_utils.squeeze_batch_dim(network.initial_state(1)),
      }
    else:
      extra_spec = ()

    reverb_replay = replay.make_reverb_prioritized_sequence_replay(
        environment_spec=environment_spec,
        extra_spec=extra_spec,
        prefetch_size=prefetch_size,
        batch_size=batch_size,
        max_replay_size=max_replay_size,
        min_replay_size=1,
        priority_exponent=priority_exponent,
        # discount=discount,
        burn_in_length=burn_in_length,
        # function internally updates sequence length
        sequence_length=trace_length,
        sequence_period=replay_period,
    )
    self._server = reverb_replay.server

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adam(learning_rate),
    )
    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(seed))

    # The learner updates the parameters (and initializes them).
    loss_fn = losses.R2D2Learning(
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        burn_in_length=burn_in_length,
        max_replay_size=max_replay_size,
        store_lstm_state=store_lstm_state,
        max_priority_weight=max_priority_weight,
    )
    learner = learning_lib.SGDLearner(
        network=network,
        loss_fn=loss_fn,
        data_iterator=reverb_replay.data_iterator,
        optimizer=optimizer,
        target_update_period=target_update_period,
        random_key=key_learner,
        replay_client=reverb_replay.client,
    )

    # -----------------------
    # define policy
    # -----------------------
    # The actor selects actions according to the policy.
    def policy(params: networks_lib.Params,
               key: jnp.ndarray,
               inputs: observation_action_reward.OAR,
               state: hk.LSTMState) -> Tuple[jnp.ndarray, hk.LSTMState]:
      action_values, next_state = network.apply(params, inputs, state)
      actions = rlax.epsilon_greedy(epsilon).sample(key, action_values)
      return actions, next_state

    # -----------------------
    # initialize actor
    # -----------------------
    rng = hk.PRNGSequence(key_actor)
    params = learner._state.params
    actor_batch_size = 1
    initial_state = network.initial_state(params, actor_batch_size)
    actor_core = actor_core_lib.batched_recurrent_to_actor_core(
      policy,
      initial_state,
      extras_recurrent_state=store_lstm_state)
    variable_client = variable_utils.VariableClient(learner, '')
    actor = actors.GenericActor(
        actor_core, key_actor, variable_client, reverb_replay.adder)

    # -----------------------
    # settings
    # -----------------------
    min_observations = replay_period * max(batch_size, min_replay_size)
    observations_per_step = (
        float(replay_period * batch_size) / samples_per_insert)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=min_observations,
        observations_per_step=observations_per_step,
    )


