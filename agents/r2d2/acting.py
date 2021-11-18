"""R2D2 actor implementation."""

from typing import Optional

from acme import adders
from acme import core
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

import rlax
from agents.r2d2 import networks


class R2D2Actor(core.Actor):
  """A recurrent actor."""

  _state: hk.LSTMState
  _prev_state: hk.LSTMState

  def __init__(
      self,
      forward_fn: networks.ValueFn,
      initial_state_init_fn: networks.RecurrentStateInitFn,
      initial_state_fn: networks.RecurrentStateFn,
      rng: hk.PRNGSequence,
      epsilon: int,
      variable_client: Optional[variable_utils.VariableClient] = None,
      adder: Optional[adders.Adder] = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._forward = forward_fn
    self._reset_fn_or_none = getattr(forward_fn, 'reset', None)
    self._rng = rng
    self._epsilon = epsilon

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    if self._variable_client is not None:
      self._variable_client.update_and_wait()

    params = initial_state_init_fn(next(self._rng))
    self._initial_state = initial_state_fn(params)

  def select_action(self, observation: networks.Observation) -> networks.Action:

    if self._state is None:
      self._state = self._initial_state

    # Forward.
    action_values, new_state = self._forward(self._params,
                                             observation,
                                             self._state)

    # self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state

    action = rlax.epsilon_greedy(self._epsilon).sample(next(self._rng), action_values)

    return action.astype(jnp.int32)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

    # Reset state of inference functions that employ stateful wrappers (eg. BIT)
    # at the start of the episode.
    if self._reset_fn_or_none is not None:
      self._reset_fn_or_none()

  def observe(
      self,
      action: networks.Action,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return

    extras = {'core_state': self._prev_state}
    self._adder.add(action, next_timestep, extras)

  def update(self, wait: bool = False):
    if self._variable_client is not None:
      self._variable_client.update(wait)

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params