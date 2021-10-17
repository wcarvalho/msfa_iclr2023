import dataclasses

from typing import Callable

from acme.jax import networks
import haiku as hk
import jax.numpy as jnp


# Only simple observations & discrete action spaces for now.
Observation = jnp.ndarray
Action = int
ValueInitFn = Callable[[networks.PRNGKey, Observation, hk.LSTMState],
                             networks.Params]
ValueFn = Callable[[networks.Params, Observation, hk.LSTMState],
                         networks.Value]
RecurrentStateInitFn = Callable[[networks.PRNGKey], networks.Params]
RecurrentStateFn = Callable[[networks.Params], hk.LSTMState]


@dataclasses.dataclass
class R2D2Network:
  """Pure functions representing R2D2's recurrent network components.

  Attributes:
    forward_fn: Selects next action using the network at the given recurrent
      state.
    unroll_init_fn: Initializes params for unroll_fn.
    unroll_fn: Applies the unrolled network to a sequence of observations, for
      learning.
    initial_state_init_fn: Initializes params for initial_state_fn.
    initial_state_fn: Recurrent state at the beginning of an episode.
  """
  apply: types.ValueFn
  unroll: types.ValueFn
  init: types.ValueInitFn

  # forward_fn: types.ValueFn
  # unroll_fn: types.ValueFn
  # unroll_init_fn: types.ValueInitFn
  # initial_state_init_fn: types.RecurrentStateInitFn
  # initial_state_fn: types.RecurrentStateFn
