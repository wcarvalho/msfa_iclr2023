from typing import Optional

from acme import types
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import networks as r2d2_networks_lib
from acme.jax import networks as networks_lib
from acme.jax import utils

import haiku as hk
import jax
import rlax

from agents.usfa import config as usfa_config

def make_usfa_networks(batch_size, env_spec, NetworkCls, NetKwargs):
  """Builds default R2D2 networks."""

  # ======================================================
  # Functions for use
  # ======================================================
  def forward_fn(x, s, k):
    model = NetworkCls(**NetKwargs)
    return model(x, s, k)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = NetworkCls(**NetKwargs)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state, key):
    model = NetworkCls(**NetKwargs)
    return model.unroll(inputs, state, key)

  # Make networks purely functional.
  forward_hk = hk.transform(forward_fn)
  initial_state_hk = hk.transform(initial_state_fn)
  unroll_hk = hk.transform(unroll_fn)

  # ======================================================
  # Define networks init functions.
  # ======================================================
  def initial_state_init_fn(rng, batch_size):
    return initial_state_hk.init(rng, batch_size)
  dummy_obs_batch = utils.tile_nested(
      utils.zeros_like(env_spec.observations), batch_size)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs_batch)
  def unroll_init_fn(rng, initial_state):
    rng, key_init = jax.random.split(rng)
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state, key_init)


  # Make FeedForwardNetworks.
  forward = networks_lib.FeedForwardNetwork(
      init=forward_hk.init, apply=forward_hk.apply)
  unroll = networks_lib.FeedForwardNetwork(
      init=unroll_init_fn, apply=unroll_hk.apply)
  initial_state = networks_lib.FeedForwardNetwork(
      init=initial_state_init_fn, apply=initial_state_hk.apply)
  return r2d2.R2D2Networks(
      forward=forward, unroll=unroll, initial_state=initial_state)



def make_behavior_policy(
    networks: r2d2.R2D2Networks,
    config: usfa_config.Config,
    evaluation: bool = False) -> r2d2_networks_lib.EpsilonRecurrentPolicy:
  """Selects action according to the policy."""

  def behavior_policy(
                      params: networks_lib.Params,
                      key: networks_lib.PRNGKey,
                      observation: types.NestedArray,
                      core_state: types.NestedArray,
                      epsilon):
    key, key_init = jax.random.split(key)
    q_values, core_state = networks.forward.apply(
        params, key, observation, core_state, key_init)
    epsilon = config.evaluation_epsilon if evaluation else epsilon
    return rlax.epsilon_greedy(epsilon).sample(key, q_values), core_state

  return behavior_policy
