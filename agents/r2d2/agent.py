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

from acme import specs
from acme.agents import agent
from acme.agents import replay
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax.dqn import learning_lib
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.jax import utils as jax_utils

import jax
import jax.numpy as jnp
import optax
import rlax

from agents.r2d2 import losses
from agents.r2d2 import config
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
      burn_in_length: int,
      trace_length: int,
      replay_period: int,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 0.5,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
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
        batch_size=batch_size,
        max_replay_size=max_replay_size,
        min_replay_size=min_replay_size,
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
    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    # The learner updates the parameters (and initializes them).
    sequence_length = burn_in_length + trace_length + 1
    loss_fn = losses.R2D2Learning(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=burn_in_length,
        sequence_length=sequence_length,
        max_replay_size=max_replay_size,
        store_lstm_state=store_lstm_state,
        max_priority_weight=max_priority_weight,
    )
    learner = learning_lib.SGDLearner(
        network=network,
        loss_fn=loss_fn,
        data_iterator=reverb_replay.data_iterator,
        optimizer=optimizer,
        target_update_period=config.target_update_period,
        random_key=key_learner,
        replay_client=reverb_replay.client,
    )

    # The actor selects actions according to the policy.
    def policy(params: networks_lib.Params, key: jnp.ndarray,
               observation: jnp.ndarray) -> jnp.ndarray:
      action_values = network.apply(params, observation)
      return rlax.epsilon_greedy(config.epsilon).sample(key, action_values)
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
    variable_client = variable_utils.VariableClient(learner, '')
    actor = actors.GenericActor(
        actor_core, key_actor, variable_client, reverb_replay.adder)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(config.batch_size, config.min_replay_size),
        observations_per_step=config.batch_size / config.samples_per_insert,
    )


