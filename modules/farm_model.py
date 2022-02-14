import jax
import jax.numpy as jnp
import haiku as hk
from modules.basic_archs import AuxilliaryTask

TIME_AXIS=0
BATCH_AXIS=1
MODULE_AXIS=2
FEATURE_AXIS=3

from utils import data as data_utils

class FarmModel(AuxilliaryTask):
  """docstring for FarmModel"""
  def __init__(self, *args, num_actions, **kwargs):
    super(FarmModel, self).__init__(unroll_only=True, timeseries=True)
    self.model = hk.nets.MLP(*args, **kwargs)
    self.num_actions = num_actions


  def __call__(self, inputs, memory_out, predictions, **kwargs):

    module_states = memory_out[:-1]  # [T, B, N, D]
    T, B, N, D = module_states.shape

    # [T, B, A]
    chosen_actions = jax.nn.one_hot(inputs.action[1:],
      num_classes=self.num_actions)
    # [T, B, N, A]
    chosen_actions = data_utils.expand_tile_dim(chosen_actions, axis=-2, size=N)

    model_input = jnp.concatenate((module_states, chosen_actions), axis=-1)
    model_outputs = hk.BatchApply(self.model, num_dims=3)(model_input)

    return {'model_outputs' : model_outputs}

class FarmCumulants(AuxilliaryTask):
  """docstring for FarmCumulants"""
  def __init__(self, *args, cumtype='sum', normalize=True, **kwargs):
    super(FarmCumulants, self).__init__(unroll_only=True, timeseries=True)
    self.cumulant_fn = hk.nets.MLP(*args, **kwargs)
    cumtype = cumtype.lower()
    assert cumtype in ['sum', 'weighted', 'concat']
    self.cumtype = cumtype
    self.normalize = normalize

  def __call__(self, memory_out, predictions, **kwargs):

    states = memory_out[:-1]  # [T, B, N, D]
    next_states = memory_out[1:]  # [T, B, N, D]

    delta = next_states - states
    if self.normalize:
      delta = delta / jnp.linalg.norm(delta, axis=-1, keepdims=True)

    if self.cumtype == "sum":
      delta = delta.sum(axis=MODULE_AXIS)
    elif self.cumtype == "concat":
      delta = delta.reshape(*delta.shape[:2], -1)
    elif self.cumtype == "weighted":
      raise NotImplementedError

    return {'cumulants' : self.cumulant_fn(delta)}
