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

def overlapping(dict1, dict2):
  return set(dict1.keys()).intersection(dict2.keys())

class AttrDict(dict):
    def __getattr__(self, attr):
      try:
        return self[attr]
      except Exception as e:
        getattr(self, attr)

    def __setattr__(self, attr, value):
      self[attr] = value

class BasicRecurrent(hk.Module):
  """docstring for BasicRecurrent"""
  def __init__(self,
    vision : hk.Module,
    memory : hk.Module,
    prediction : hk.Module,
    evaluation : hk.Module=None,
    aux_tasks: Union[Callable, Sequence[Callable]]=None,
    inputs_prep_fn : Callable=None,
    vision_prep_fn : Callable=None,
    memory_prep_fn : Callable=None,
    memory_proc_fn : Callable=None,
    prediction_prep_fn : Callable=None,
    evaluation_prep_fn : Callable=None,
    PredCls: NamedTuple=None,
    ):
    super(BasicRecurrent, self).__init__()
    self.vision = vision
    self.memory = memory
    self.prediction = prediction
    if evaluation is None:
      evaluation = prediction
    self.evaluation = evaluation

    self.inputs_prep_fn = inputs_prep_fn
    self.vision_prep_fn = vision_prep_fn
    self.memory_prep_fn = memory_prep_fn
    self.memory_proc_fn = memory_proc_fn
    self.prediction_prep_fn = prediction_prep_fn
    if evaluation_prep_fn is None:
        evaluation_prep_fn = prediction_prep_fn
    self.evaluation_prep_fn = evaluation_prep_fn
    self.PredCls = PredCls

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
    return self.forward(inputs, state, key, setting="train")

  def unroll(
      self,
      inputs: Any,  # [T, B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Any, hk.LSTMState]:
    return self.forward(inputs, state, key, setting="unroll")

  def evaluate(
      self,
      inputs: Any,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Any, hk.LSTMState]:
    return self.forward(inputs, state, key, setting="evaluate")

  def forward(
      self,
      inputs: Any,
      state: hk.LSTMState,
      key: networks_lib.PRNGKey,
      setting='unroll',
    ) -> Tuple[Any, hk.LSTMState]:
    """
    1. process inputs
    2. vision function
    3. memory function
    4. prediction function
    Efficient unroll that applies torso, core, and output in one pass.
    """
    assert setting in ['unroll', 'train', 'evaluate']
    unroll = setting == "unroll"
    batchfn = hk.BatchApply if unroll else lambda x:x

    all_preds = AttrDict()
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
      memory_out, new_state = hk.static_unroll(self.memory, memory_input, state)
    else:
      memory_out, new_state = self.memory(memory_input, state)

    if self.memory_proc_fn:
      memory_out = batchfn(self.memory_proc_fn)(memory_out)

    all_preds['memory_out'] = memory_out
  

    # ======================================================
    # Predictions
    # ======================================================

    if setting == "train" or setting == "unroll":
      if self.prediction_prep_fn:
        prediction_input = self.prediction_prep_fn(
          inputs=inputs, obs=obs, memory_out=memory_out)
      else:
        prediction_input = memory_out
      pred_fun = functools.partial(self.prediction, key=key)
    elif setting == "evaluate":
      if self.evaluation_prep_fn:
        prediction_input = self.evaluation_prep_fn(
            inputs=inputs, obs=obs, memory_out=memory_out)
      else:
        prediction_input = memory_out
      pred_fun = functools.partial(self.evaluation, key=key)

    predictions = batchfn(pred_fun)(prediction_input)
    predictions = predictions._asdict()
    overlapping_keys = overlapping(predictions, all_preds)
    assert len(overlapping_keys) == 0, "overwriting!"
    all_preds.update(predictions)

    # ======================================================
    # Auxiliary Tasks
    # ======================================================
    forward=setting in ['train', 'evaluate']
    if self.aux_tasks:
      for aux_task in self.aux_tasks:
        # -----------------------
        # does this aux task only occur during unroll (not forward?)
        # -----------------------
        unroll_only = getattr(aux_task, 'unroll_only', False)
        if unroll_only and forward: continue

        # -----------------------
        # if aux task is for time-series or during forward, no BatchApply
        # -----------------------
        aux_for_timeseries = getattr(aux_task, 'timeseries', False)
        if aux_for_timeseries or forward:
          batchfn = lambda x:x
        else:
          batchfn = hk.BatchApply 

        aux_pred = batchfn(aux_task)(
          inputs=inputs,
          obs=obs,
          memory_out=memory_out,
          predictions=predictions
          )
        overlapping_keys = overlapping(aux_pred, all_preds)
        assert len(overlapping_keys) == 0, "replacing values?"
        all_preds.update(aux_pred)

    if self.PredCls is not None:
      all_preds = self.PredCls(**all_preds)
    else:
      # ONLY use this during creation. Constantly creating namedtuples can cause a memory leak.
      PredCls = collections.namedtuple('Predictions', all_preds.keys())
      all_preds = PredCls(**all_preds)

    return all_preds, new_state

