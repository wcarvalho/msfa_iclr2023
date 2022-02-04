import dataclasses

from typing import Callable, Optional, Tuple, NamedTuple

from acme import types
from acme.jax import networks as networks_lib
from acme.jax.networks import base
from acme.jax.networks import embedding
from acme.jax.networks import duelling
from acme.wrappers import observation_action_reward
import functools

import haiku as hk
import jax
import jax.numpy as jnp

class Network(hk.RNNCore):
  """docstring for network"""
  def __init__(self, input_process_fn, vision_net, memory_net, prediction_net, vision_rng=False, memory_rng=False, prediction_rng=False):
    super(network, self).__init__()
    self.input_process_fn = input_process_fn
    self.vision_net = vision_net
    self.memory_net = memory_net
    self.prediction_net = prediction_net
    self.vision_rng = vision_rng
    self.memory_rng = memory_rng
    self.prediction_rng = prediction_rng



  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[Predictions, hk.LSTMState]:

    vision_input, memory_input, prediction_input = self.input_process_fn(inputs)
    obs = self.vision_net(vision_input)  # [B, D+A+1]

    memory_input = jnp.concatenate([obs, memory_input], axis=-1)
    core_outputs, new_state = self._core(memory_input, state, key)

    prediction_input = jnp.concatenate((core_outputs, prediction_input), axis=-1)
    prediction = self.prediction_net(prediction_input, key)
    return prediction, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory_net.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[Predictions, hk.LSTMState]:
    del key
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    vision_input, memory_input, prediction_input = self.input_process_fn(inputs)
    obs = hk.BatchApply(self._embed)(vision_input)  # [T, B, D+A+1]

    memory_input = jnp.concatenate([obs, memory_input], axis=-1)
    core_outputs, new_states = hk.static_unroll(self._core, memory_input, state)

    prediction_input = jnp.concatenate((core_outputs, prediction_input), axis=-1)
    prediction = hk.BatchApply(self.prediction_net)(prediction_input)  # [T, B, A]
    return prediction, new_states
