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

"""ActorCore interface definition."""

import dataclasses
from typing import Callable, Generic, Mapping, Tuple, TypeVar, Union

from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.types import PRNGKey


import chex
import jax
import jax.numpy as jnp

from acme.agents.jax import actor_core as actor_core_lib

RecurrentState = TypeVar('RecurrentState')

RecurrentPolicy = Callable[[
    networks_lib.Params, PRNGKey, networks_lib
    .Observation, RecurrentState
], Tuple[networks_lib.Action, RecurrentState]]

ActorCore = actor_core_lib.ActorCore

@chex.dataclass(frozen=True, mappable_dataclass=False)
class SimpleActorCoreRecurrentState(Generic[RecurrentState]):
  rng: PRNGKey
  recurrent_state: RecurrentState


def batched_recurrent_to_actor_core(
    recurrent_policy: RecurrentPolicy,
    initial_core_state: RecurrentState,
    extras_recurrent_state: bool = True,
) -> ActorCore[SimpleActorCoreRecurrentState[RecurrentState], Mapping[
    str, jnp.ndarray]]:
  """Returns ActorCore for a recurrent policy."""
  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: SimpleActorCoreRecurrentState[RecurrentState]):
    # TODO: Make JAX Actor work with batched or unbatched inputs.
    rng = state.rng
    rng, policy_rng = jax.random.split(rng)
    observation = utils.add_batch_dim(observation)
    recurrent_state = utils.add_batch_dim(state.recurrent_state)
    action, new_recurrent_state = utils.squeeze_batch_dim(recurrent_policy(
        params, policy_rng, observation, recurrent_state))
    return action, SimpleActorCoreRecurrentState(rng, new_recurrent_state)

  initial_core_state = utils.squeeze_batch_dim(initial_core_state)
  def init(rng: PRNGKey) -> SimpleActorCoreRecurrentState[RecurrentState]:
    return SimpleActorCoreRecurrentState(rng, initial_core_state)

  def get_extras(
      state: SimpleActorCoreRecurrentState[RecurrentState]
  ) -> Mapping[str, jnp.ndarray]:
    if extras_recurrent_state:
      return {'core_state': state.recurrent_state}
    else:
      return {}

  return ActorCore(init=init, select_action=select_action,
                   get_extras=get_extras)
