import dataclasses

from typing import Callable

from acme.jax import networks
from acme.jax import types

import haiku as hk
import jax.numpy as jnp


# Only simple observations & discrete action spaces for now.
Observation = jnp.ndarray
# Action = int
ValueInitFn = Callable[[networks.PRNGKey, Observation, hk.LSTMState],
                             networks.Params]
ValueFn = Callable[[networks.Params, Observation, hk.LSTMState],
                         networks.Value]
RecurrentStateInitFn = Callable[[networks.PRNGKey], networks.Params]
# RecurrentStateFn = Callable[[networks.Params], hk.LSTMState]


@dataclasses.dataclass
class R2D2Network(networks.FeedForwardNetwork):
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
  apply: ValueFn
  unroll: ValueFn
  initial_state: RecurrentStateInitFn
