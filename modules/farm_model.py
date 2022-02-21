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
  def __init__(self, out_dim=0, hidden_size=0, cumtype='sum', normalize_delta=True, normalize_cumulants=True, **kwargs):
    super(FarmCumulants, self).__init__(
      unroll_only=True, timeseries=True)

    if hidden_size:
      layers = [hidden_size, out_dim]
    else:
      layers = [out_dim]

    if out_dim > 0:
      self.cumulant_fn = hk.nets.MLP(layers)
    else:
      self.cumulant_fn = lambda x:x

    self.out_dim = out_dim
    cumtype = cumtype.lower()
    assert cumtype in ['sum', 'weighted', 'concat']
    self.cumtype = cumtype
    self.normalize_delta = normalize_delta
    self.normalize_cumulants = normalize_cumulants

  def __call__(self, memory_out, predictions, **kwargs):

    states = memory_out[:-1]  # [T, B, N, D]
    next_states = memory_out[1:]  # [T, B, N, D]

    delta = next_states - states
    if self.normalize_delta:
      delta = delta / (1e-5+jnp.linalg.norm(delta, axis=-1, keepdims=True))

    if self.cumtype == "sum":
      delta = delta.sum(axis=MODULE_AXIS)
      # assert delta.shape[-1] == states.shape[-1]
    elif self.cumtype == "concat":
      delta = delta.reshape(*delta.shape[:2], -1)
      # assert delta.shape[-1] == states.shape[-1]*states.shape[-2]
    elif self.cumtype == "weighted":
      raise NotImplementedError

    # assert len(delta.shape) == 3, "should be T x B x D"
    cumulants = self.cumulant_fn(delta)
    if self.normalize_cumulants:
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    return {'cumulants' : cumulants}
