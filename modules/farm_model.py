import jax
import jax.numpy as jnp
import haiku as hk
from modules.basic_archs import AuxilliaryTask

TIME_AXIS=0
BATCH_AXIS=1
MODULE_AXIS=2
FEATURE_AXIS=3

from utils import data as data_utils
from utils.vmap import batch_multihead

class FarmModel(AuxilliaryTask):
  """docstring for FarmModel"""
  def __init__(self, *args, num_actions, **kwargs):
    super(FarmModel, self).__init__(unroll_only=True, timeseries=True)
    self.model = hk.nets.MLP(*args, **kwargs)
    self.num_actions = num_actions


  def __call__(self, inputs, memory_out, predictions, **kwargs):
    """Apply model to each module memory.
    """
    module_states = memory_out[:-1]  # [T, B, M, D]
    T, B, M, D = module_states.shape

    # [T, B, A]
    chosen_actions = jax.nn.one_hot(inputs.action[1:],
      num_classes=self.num_actions)
    # [T, B, M, A]
    chosen_actions = data_utils.expand_tile_dim(chosen_actions, axis=2, size=M)

    model_input = jnp.concatenate((module_states, chosen_actions), axis=-1)
    model_outputs = hk.BatchApply(self.model, num_dims=3)(model_input)

    return {'model_outputs' : model_outputs}

class FarmCumulants(AuxilliaryTask):
  """docstring for FarmCumulants"""
  def __init__(self,
    module_cumulants=0,
    hidden_size=0,
    aggregation='sum',
    construction='timestep',
    normalize_delta=True,
    normalize_cumulants=True,
    **kwargs):
    """Summary
    
    Args:
        module_cumulants (int, optional): Description
        hidden_size (int, optional): Description
        aggregation (str, optional): how to aggregate modules for cumulant
        use_delta (bool, optional): whether to use delta between states as cumulant
        normalize_delta (bool, optional): Description
        normalize_cumulants (bool, optional): Description
        **kwargs: Description
    """
    super(FarmCumulants, self).__init__(
      unroll_only=True, timeseries=True)
    if hidden_size:
      layers = [hidden_size, module_cumulants]
    else:
      layers = [module_cumulants]

    if module_cumulants > 0:
      self.cumulant_fn_factory = lambda: hk.nets.MLP(layers)

    self.module_cumulants = module_cumulants
    self.hidden_size = hidden_size
    aggregation = aggregation.lower()
    assert aggregation in ['sum', 'weighted', 'concat']
    self.aggregation = aggregation

    self.normalize_delta = normalize_delta
    self.normalize_cumulants = normalize_cumulants

    self.construction = construction.lower()
    assert self.construction in ['timestep', 'delta', 'concat']

  def __call__(self, memory_out, predictions, **kwargs):

    if self.construction == 'delta':
      states = memory_out[:-1]  # [T, B, M, D]
      next_states = memory_out[1:]  # [T, B, M, D]

      cumulants = next_states - states
      if self.normalize_delta:
        cumulants = cumulants / (1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    elif self.construction == 'concat':
      states = memory_out[:-1]  # [T, B, M, D]
      next_states = memory_out[1:]  # [T, B, M, D]
      cumulants = jnp.concatenate((next_states, states), axis=-1)

    elif self.construction == 'timestep':
      cumulants = memory_out

    if self.aggregation == "sum":
      cumulants = cumulants.sum(axis=MODULE_AXIS)
    elif self.aggregation == "concat":
      cumulants = cumulants.reshape(*cumulants.shape[:2], -1)
    elif self.aggregation == "weighted":
      raise NotImplementedError

    if self.module_cumulants > 0:
      cumulants = self.cumulant_fn_factory()(cumulants)

    if self.normalize_cumulants:
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))

    return {'cumulants' : cumulants}



class FarmIndependentCumulants(FarmCumulants):
  """Each FARM module predicts its own set of cumulants"""

  def __call__(self, memory_out, predictions, **kwargs):

    if self.construction == 'delta':
      states = memory_out[:-1]  # [T, B, M, D]
      next_states = memory_out[1:]  # [T, B, M, D]

      cumulants = next_states - states
      if self.normalize_delta:
        cumulants = cumulants / (1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    elif self.construction == 'concat':
      states = memory_out[:-1]  # [T, B, M, D]
      next_states = memory_out[1:]  # [T, B, M, D]
      cumulants = jnp.concatenate((next_states, states), axis=-1)

    elif self.construction == 'timestep':
      cumulants = memory_out

    _cumulants = []
    M = cumulants.shape[2]
    for idx in range(M):
      c = hk.BatchApply(self.cumulant_fn_factory())(cumulants[:,:, idx])
      _cumulants.append(c)
    # M x [T, B, D] --> [T, B, M*D]
    cumulants = jnp.concatenate(_cumulants, axis=2)

    if self.normalize_cumulants:
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))

    return {'cumulants' : cumulants}
