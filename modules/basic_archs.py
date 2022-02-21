from typing import Any, Callable, Optional, Tuple, Sequence, NamedTuple, Union

from acme.jax import networks as networks_lib

import collections
import functools
import jax
import haiku as hk

from utils import data as data_utils

class AuxilliaryTask(hk.Module):
  """docstring for AuxilliaryTask"""
  def __init__(self, unroll_only=False, timeseries=False):
    super(AuxilliaryTask, self).__init__()
    self.unroll_only = unroll_only
    self.timeseries = timeseries


class BasicRecurrent(hk.Module):
  """docstring for BasicRecurrent"""
  def __init__(self,
    vision : hk.Module,
    memory : hk.Module,
    prediction : hk.Module,
    aux_tasks: Union[Callable, Sequence[Callable]]=None,
    inputs_prep_fn : Callable=None,
    vision_prep_fn : Callable=None,
    memory_prep_fn : Callable=None,
    memory_proc_fn : Callable=None,
    prediction_prep_fn : Callable=None,
    ):
    super(BasicRecurrent, self).__init__()
    self.vision = vision
    self.memory = memory
    self.prediction = prediction

    self.inputs_prep_fn = inputs_prep_fn
    self.vision_prep_fn = vision_prep_fn
    self.memory_prep_fn = memory_prep_fn
    self.memory_proc_fn = memory_proc_fn
    self.prediction_prep_fn = prediction_prep_fn

    # -----------------------
    # auxilliary tasks
    # -----------------------
    # if have auxiliary tasks and only 1, make into list
    if aux_tasks is not None: 
      if not isinstance(aux_tasks, list): aux_tasks = [aux_tasks]
    self.aux_tasks = aux_tasks

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

    # ======================================================
    # Vision
    # ======================================================
    if self.vision_prep_fn:
      vision_input = self.vision_prep_fn(inputs=inputs)
    else:
      vision_input = inputs
    obs = self.vision(vision_input)

    # ======================================================
    # Memory
    # ======================================================
    if self.memory_prep_fn:
      memory_input = self.memory_prep_fn(inputs=inputs, obs=obs)
    else:
      memory_input = obs
    memory_out, new_state = self.memory(memory_input, state)
    if self.memory_proc_fn:
      memory_out = self.memory_proc_fn(memory_out)

    # ======================================================
    # Predictions
    # ======================================================
    if self.prediction_prep_fn:
      prediction_input = self.prediction_prep_fn(
        inputs=inputs, obs=obs, memory_out=memory_out)
    else:
      prediction_input = memory_out
    predictions = self.prediction(prediction_input, key=key)

    # ======================================================
    # Auxiliary Tasks
    # ======================================================
    if self.aux_tasks:
      predictions = self.auxilliary_tasks(
        inputs=inputs,
        obs=obs,
        memory_out=memory_out,
        predictions=predictions,
        unroll=False)

    return predictions, new_state

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

    # ======================================================
    # Vision
    # ======================================================
    if self.vision_prep_fn:
      vision_input = self.vision_prep_fn(inputs=inputs)
    else:
      vision_input = inputs
    obs = hk.BatchApply(self.vision)(vision_input)

    # ======================================================
    # Memory
    # ======================================================
    if self.memory_prep_fn:
      memory_input = self.memory_prep_fn(inputs=inputs, obs=obs)
    else:
      memory_input = obs
    memory_out, new_states = hk.static_unroll(self.memory, memory_input, state)
    if self.memory_proc_fn:
      memory_out = self.memory_proc_fn(memory_out)

    # ======================================================
    # Predictions
    # ======================================================
    if self.prediction_prep_fn:
      prediction_input = self.prediction_prep_fn(
        inputs=inputs, obs=obs, memory_out=memory_out)
    else:
      prediction_input = memory_out

    pred_fun = functools.partial(self.prediction, key=key)
    predictions = hk.BatchApply(pred_fun)(prediction_input)

    # ======================================================
    # Auxiliary Tasks
    # ======================================================
    if self.aux_tasks:
      predictions = self.auxilliary_tasks(
        inputs=inputs,
        obs=obs,
        memory_out=memory_out,
        predictions=predictions,
        unroll=True)
    return predictions, new_states

  def auxilliary_tasks(self,
    inputs,
    obs,
    memory_out,
    predictions : NamedTuple,
    unroll=True):
    all_preds = predictions._asdict()
    all_preds['obs'] = obs
    all_preds['memory_out'] = memory_out

    inference=not unroll
    for aux_task in self.aux_tasks:
      # -----------------------
      # does this aux task only occur during unroll (not inference?)
      # -----------------------
      unroll_only = getattr(aux_task, 'unroll_only', False)
      if unroll_only and inference: continue

      # -----------------------
      # if aux task is for time-series or during inference, no BatchApply
      # -----------------------
      aux_for_timeseries = getattr(aux_task, 'timeseries', False)
      if aux_for_timeseries or inference:
        batchfn = lambda x:x
      else:
        batchfn = hk.BatchApply 

      aux_pred = batchfn(aux_task)(
        inputs=inputs,
        obs=obs,
        memory_out=memory_out,
        predictions=predictions
        )
      overlapping_keys = set(aux_pred.keys()).intersection(all_preds.keys())
      assert len(overlapping_keys) == 0, "replacing values?"
      all_preds.update(aux_pred)

    Predictions = collections.namedtuple('Predictions', all_preds.keys())
    predictions = Predictions(**all_preds)

    return predictions


class BasicRecurrentUnified(hk.Module):
  """docstring for BasicRecurrentUnified"""
  def __init__(self,
    vision : hk.Module,
    memory : hk.Module,
    prediction : hk.Module,
    aux_tasks: Union[Callable, Sequence[Callable]]=None,
    inputs_prep_fn : Callable=None,
    vision_prep_fn : Callable=None,
    memory_prep_fn : Callable=None,
    memory_proc_fn : Callable=None,
    prediction_prep_fn : Callable=None,
    ):
    super(BasicRecurrentUnified, self).__init__()
    self.vision = vision
    self.memory = memory
    self.prediction = prediction

    self.inputs_prep_fn = inputs_prep_fn
    self.vision_prep_fn = vision_prep_fn
    self.memory_prep_fn = memory_prep_fn
    self.memory_proc_fn = memory_proc_fn
    self.prediction_prep_fn = prediction_prep_fn

    # -----------------------
    # auxilliary tasks
    # -----------------------
    # if have auxiliary tasks and only 1, make into list
    if aux_tasks is not None: 
      if not isinstance(aux_tasks, list): aux_tasks = [aux_tasks]
    self.aux_tasks = aux_tasks

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory.initial_state(batch_size)

  def __call__(
      self,
      inputs : Any,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Any, hk.LSTMState]:
    return self.forward(inputs, state, key, unroll=False)

  def unroll(
      self,
      inputs: Any,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Any, hk.LSTMState]:
    return self.forward(inputs, state, key, unroll=True)


  def forward(
      self,
      inputs: Any,
      state: hk.LSTMState,
      key: networks_lib.PRNGKey,
      unroll=False,
    ) -> Tuple[Any, hk.LSTMState]:
    """
    1. process inputs
    2. vision function
    3. memory function
    4. prediction function
    Efficient unroll that applies torso, core, and output in one pass.
    """

    batchfn = hk.BatchApply if unroll else lambda x:x

    all_preds = {}
    if self.inputs_prep_fn:
      inputs = self.inputs_prep_fn(inputs)

    # ======================================================
    # Vision
    # ======================================================
    if self.vision_prep_fn:
      vision_input = self.vision_prep_fn(inputs=inputs)
    else:
      vision_input = inputs
    obs = batchfn(self.vision)(vision_input)
    all_preds['obs'] = obs

    # ======================================================
    # Memory
    # ======================================================
    if self.memory_prep_fn:
      memory_input = self.memory_prep_fn(inputs=inputs, obs=obs)
    else:
      memory_input = obs

    if unroll:
      memory_out, new_states = hk.static_unroll(self.memory, memory_input, state)
    else:
      memory_out, new_states = self.memory(memory_input, state)

    if self.memory_proc_fn:
      memory_out = self.memory_proc_fn(memory_out)
    all_preds['memory_out'] = memory_out


    # ======================================================
    # Predictions
    # ======================================================
    if self.prediction_prep_fn:
      prediction_input = self.prediction_prep_fn(
        inputs=inputs, obs=obs, memory_out=memory_out)
    else:
      prediction_input = memory_out

    pred_fun = functools.partial(self.prediction, key=key)
    predictions = batchfn(pred_fun)(prediction_input)
    all_preds.update(predictions._asdict())

    # ======================================================
    # Auxiliary Tasks
    # ======================================================
    inference=not unroll
    for aux_task in self.aux_tasks:
      # -----------------------
      # does this aux task only occur during unroll (not inference?)
      # -----------------------
      unroll_only = getattr(aux_task, 'unroll_only', False)
      if unroll_only and inference: continue

      # -----------------------
      # if aux task is for time-series or during inference, no BatchApply
      # -----------------------
      aux_for_timeseries = getattr(aux_task, 'timeseries', False)
      if aux_for_timeseries or inference:
        batchfn = lambda x:x
      else:
        batchfn = hk.BatchApply 

      aux_pred = batchfn(aux_task)(
        inputs=inputs,
        obs=obs,
        memory_out=memory_out,
        predictions=predictions
        )
      overlapping_keys = set(aux_pred.keys()).intersection(all_preds.keys())
      assert len(overlapping_keys) == 0, "replacing values?"
      all_preds.update(aux_pred)

    Predictions = collections.namedtuple('Predictions', all_preds.keys())
    all_preds = Predictions(**all_preds)

    return all_preds, new_states

