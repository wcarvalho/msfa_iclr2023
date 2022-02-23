from typing import Optional

from acme import types
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import networks as r2d2_networks_lib
from acme.jax import networks as networks_lib
from acme.jax import utils

import haiku as hk
import jax
import rlax
import numpy as np

from agents.td_agent.types import TDNetworkFns, Predictions
from agents.td_agent.configs import R2D1Config

def make_networks(batch_size : int, env_spec, NetworkCls, NetKwargs, eval_network: bool=False):
  """Builds USF networks.
  
  Args:
      batch_size (TYPE): Description
      env_spec (TYPE): Description
      NetworkCls (TYPE): Description
      NetKwargs (TYPE): Description
      eval_network (bool, optional): whether to make seperate evaluation network
  
  Returns:
      TYPE: Description
  """

  # ======================================================
  # Functions for use
  # ======================================================
  def forward_fn(x, s, k: Optional[int]=None):
    model = NetworkCls(**NetKwargs)
    return model(x, s, k)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = NetworkCls(**NetKwargs)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state, key: Optional[int]=None):
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
  dummy_length = 4
  dummy_obs_sequence = utils.tile_nested(dummy_obs_batch, dummy_length)

  def unroll_init_fn(rng, initial_state):
    rng, rng_init = jax.random.split(rng)
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state, rng_init)


  # Make FeedForwardNetworks.
  forward = networks_lib.FeedForwardNetwork(
      init=forward_hk.init, apply=forward_hk.apply)
  unroll = networks_lib.FeedForwardNetwork(
      init=unroll_init_fn, apply=unroll_hk.apply)
  initial_state = networks_lib.FeedForwardNetwork(
      init=initial_state_init_fn, apply=initial_state_hk.apply)

  # -----------------------
  # optional evaluation network
  # -----------------------
  evaluation=None
  if eval_network:
    def eval_fn(x, s, k: Optional[int]=None):
      model = NetworkCls(**NetKwargs)
      return model.evaluate(x, s, k)
    eval_hk = hk.transform(eval_fn)
    evaluation = networks_lib.FeedForwardNetwork(
      init=eval_hk.init, apply=eval_hk.apply)


  # ======================================================
  # create initialization function
  # ======================================================
  def init(random_key):
    random_key, key_initial_1, key_initial_2 = jax.random.split(random_key, 3)
    initial_state_params = initial_state.init(key_initial_1, batch_size)
    initial_mem_state = initial_state.apply(initial_state_params, key_initial_2, batch_size)
    random_key, key_init = jax.random.split(random_key)
    initial_params = unroll.init(key_init, initial_mem_state)

    nparams = sum(x.size for x in jax.tree_leaves(initial_params))
    print("="*25)
    print(f"Number of params: {nparams:,}")
    print("="*25)

    return initial_params

  # this conforms to DQN, R2D2, & USFA's APIs
  return TDNetworkFns(
      init=init,
      forward=forward,
      evaluation=evaluation,
      unroll=unroll,
      initial_state=initial_state)



def make_behavior_policy(
    networks: TDNetworkFns,
    config: R2D1Config,
    evaluation: bool = False,
    ) -> r2d2_networks_lib.EpsilonRecurrentPolicy:
  """Selects action according to the policy.
  
  Args:
      networks (TDNetworkFns): Network functions
      config (R2D1Config): Config
      evaluation (bool, optional): whether evaluation policy
      network_samples (bool, optional): whether network is random
  
  Returns:
      r2d2_networks_lib.EpsilonRecurrentPolicy: epsilon-greedy policy
  """

  def behavior_policy(
                      params: networks_lib.Params,
                      key: networks_lib.PRNGKey,
                      observation: types.NestedArray,
                      core_state: types.NestedArray,
                      epsilon):
    key, key_net, key_sample = jax.random.split(key, 3)

    # -----------------------
    # if evaluating & have seperation evaluation function, use it
    # -----------------------
    if evaluation and networks.evaluation is not None:
      forward_fn = networks.evaluation.apply
    else:
      forward_fn = networks.forward.apply

    preds, core_state = forward_fn(
        params, key_net, observation, core_state, key_sample)
    epsilon = config.evaluation_epsilon if evaluation else epsilon
    return rlax.epsilon_greedy(epsilon).sample(key_net, preds.q),core_state

  return behavior_policy

