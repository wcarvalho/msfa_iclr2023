import dataclasses

from typing import Callable, Optional, Tuple, NamedTuple

from acme import types
from acme.jax import networks as networks_lib
from acme.jax.networks import base
from acme.jax.networks import embedding
from acme.jax.networks import duelling
from acme.wrappers import observation_action_reward


import haiku as hk
import jax
import jax.numpy as jnp


# Only simple observations & discrete action spaces for now.
Observation = jnp.ndarray
# Action = int

# initializations
ValueInitFn = Callable[[networks_lib.PRNGKey, Observation, hk.LSTMState],
                             networks_lib.Params]

# calling networks
RecurrentStateFn = Callable[[networks_lib.Params], hk.LSTMState]
ValueFn = Callable[[networks_lib.Params, Observation, hk.LSTMState],
                         networks_lib.Value]


@dataclasses.dataclass
class TDNetworkFns:
  """Pure functions representing recurrent network components.

  Attributes:
    init: Initializes params.
    forward: Computes Q-values using the network at the given recurrent
      state.
    unroll: Applies the unrolled network to a sequence of 
      observations, for learning.
    initial_state: Recurrent state at the beginning of an episode.
  """
  init: ValueInitFn
  forward: ValueFn
  unroll: ValueFn
  initial_state: RecurrentStateFn
