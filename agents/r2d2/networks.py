import abc

import dataclasses

from typing import Callable, Optional, Tuple

from acme import specs

from acme.agents.jax import actor_core
from acme.jax import networks
from acme.jax import types
from acme.jax import utils

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp


# Only simple observations & discrete action spaces for now.
Observation = jnp.ndarray

# initializations
RecurrentStateInitFn = Callable[[networks.PRNGKey], networks.Params]
ValueInitFn = Callable[[networks.PRNGKey, Observation, hk.LSTMState],
                             networks.Params]

# calling networks
RecurrentStateFn = Callable[[networks.Params], hk.LSTMState]
ValueFn = Callable[[networks.Params, Observation, hk.LSTMState],
                         networks.Value]


@dataclasses.dataclass
class R2D2Network:
  """Pure functions representing R2D2's recurrent network components.

  Attributes:
    init: Initializes params.
    apply: Computes Q-values using the network at the given recurrent
      state.
    unroll: Applies the unrolled network to a sequence of 
      observations, for learning.
    initial_state: Recurrent state at the beginning of an episode.
  """
  init: ValueInitFn
  init_initial_state: RecurrentStateInitFn
  apply: ValueFn
  unroll: ValueFn
  initial_state: RecurrentStateFn


def apply_policy_and_sample(
  network: R2D2Network,
  epsilon: float) -> actor_core.FeedForwardPolicy:
  """Returns the recurrent policy with epsilon-greedy exploration."""
  def apply_and_sample(params: networks.Params,
                       key: networks.PRNGKey,
                       observation: networks.Observation
                       ) -> networks.Action:
    """Returns an action for the given observation."""
    action_values = network.apply(params, observation)
    actions = rlax.epsilon_greedy(epsilon).sample(key, action_values)
    return actions.astype(jnp.int32)

  return apply_and_sample


def make_network(
    spec: specs.EnvironmentSpec,
    archCls : hk.RNNCore) -> R2D2Network:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  # -----------------------
  # Pure functions
  # -----------------------
  def apply_fn(x : jnp.ndarray, s : hk.LSTMState):
    model = archCls(num_dimensions)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = archCls(num_dimensions)
    return model.initial_state(batch_size)

  def unroll_fn(inputs : jnp.ndarray, state : hk.LSTMState):
    model = archCls(num_dimensions)
    return model.unroll(inputs, state)

  # -----------------------
  # Pure, Haiku-agnostic functions to define networks.
  # -----------------------
  apply_fn = hk.without_apply_rng(hk.transform(
      apply_fn,
      apply_rng=True))
  unroll_fn = hk.without_apply_rng(hk.transform(
      unroll_fn,
      apply_rng=True))
  initial_state_fn = hk.without_apply_rng(hk.transform(
      initial_state_fn,
      apply_rng=True))

  def init(key):
    dummy_obs = utils.zeros_like(spec.observations) # create
    dummy_obs = utils.add_batch_dim(dummy_obs)      # for batch
    dummy_obs = utils.add_batch_dim(dummy_obs)      # for time

    # TODO: params are not returned, only initial_params
    # so currently don't support learning params for intialization
    params = initial_state_fn.init(key)
    batch_size = 1
    initial_state = initial_state_fn.apply(params, batch_size)
    key, key_initial_state = jax.random.split(key)
    initial_params = unroll_fn.init(key, dummy_obs, initial_state)
    return initial_params


  return R2D2Network(
      init=init, # create params
      init_initial_state=initial_state_fn.init, # create params
      apply=apply_fn.apply, # call
      unroll=unroll_fn.apply, # unroll
      initial_state=initial_state_fn.apply, # initial_state
  )

