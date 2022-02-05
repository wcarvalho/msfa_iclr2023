from typing import Any, Callable, Optional, Tuple, NamedTuple

from acme.jax import networks as networks_lib

import functools
import jax
import haiku as hk


class BasicRecurrent(hk.Module):
  """docstring for BasicRecurrent"""
  def __init__(self,
    vision : hk.Module,
    memory : hk.Module,
    prediction : hk.Module,
    inputs_prep_fn : Callable=None,
    vision_prep_fn : Callable=None,
    memory_prep_fn : Callable=None,
    prediction_prep_fn : Callable=None,
    ):
    super(BasicRecurrent, self).__init__()
    self.vision = vision
    self.memory = memory
    self.prediction = prediction

    self.inputs_prep_fn = inputs_prep_fn
    self.vision_prep_fn = vision_prep_fn
    self.memory_prep_fn = memory_prep_fn
    self.prediction_prep_fn = prediction_prep_fn

  def __call__(
      self,
      inputs : Any,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Any, hk.LSTMState]:
    """
    1. process inputs
    2. vision function
    3. memory function
    4. prediction function
    """
    if self.inputs_prep_fn:
      inputs = self.inputs_prep_fn(inputs)

    if self.vision_prep_fn:
      vision_input = self.vision_prep_fn(inputs=inputs)
    else:
      vision_input = inputs
    obs = self.vision(vision_input)  # [B, D+A+1]

    if self.memory_prep_fn:
      memory_input = self.memory_prep_fn(inputs=inputs, obs=obs)
    else:
      memory_input = obs
    memory_out, new_state = self.memory(memory_input, state)

    if self.prediction_prep_fn:
      prediction_input = self.prediction_prep_fn(
        inputs=inputs, obs=obs, memory_out=memory_out)
    else:
      prediction_input = memory_out
    prediction = self.prediction(prediction_input, key=key)

    return prediction, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: Any,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Any, hk.LSTMState]:

    """Efficient unroll that applies torso, core, and output in one pass."""

    if self.inputs_prep_fn:
      inputs = self.inputs_prep_fn(inputs)

    if self.vision_prep_fn:
      vision_input = self.vision_prep_fn(inputs=inputs)
    else:
      vision_input = inputs
    obs = hk.BatchApply(self.vision)(vision_input)  # [B, D+A+1]

    if self.memory_prep_fn:
      memory_input = self.memory_prep_fn(inputs=inputs, obs=obs)
    else:
      memory_input = obs
    memory_out, new_states = hk.static_unroll(self.memory, memory_input, state)

    if self.prediction_prep_fn:
      prediction_input = self.prediction_prep_fn(
        inputs=inputs, obs=obs, memory_out=memory_out)
    else:
      prediction_input = memory_out

    pred_fun = functools.partial(self.prediction, key=key)
    prediction = hk.BatchApply(pred_fun)(prediction_input)

    return prediction, new_states
