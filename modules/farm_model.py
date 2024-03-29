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
  def __init__(self, output_sizes, num_actions, seperate_params=False):
    super(FarmModel, self).__init__(unroll_only=True, timeseries=True)
    self.output_sizes = output_sizes
    self.num_actions = num_actions
    self.seperate_params = seperate_params


  def __call__(self, inputs, memory_out, predictions, **kwargs):
    """Apply model to each module memory.
    """
    if hk.running_init():
      # during init, T=1
      module_states = memory_out  # [T, B, M, D]
      # [T, B, A]
      chosen_actions = jax.nn.one_hot(inputs.action,
        num_classes=self.num_actions)
    else:
      module_states = memory_out[:-1]  # [T, B, M, D]
      chosen_actions = jax.nn.one_hot(inputs.action[1:],
        num_classes=self.num_actions)

    T, B, M, D = module_states.shape


    # [T, B, M, A]
    chosen_actions = data_utils.expand_tile_dim(chosen_actions, axis=2, size=M)

    # [T, B, M, D+A]
    model_input = jnp.concatenate((module_states, chosen_actions), axis=-1)

    model_factory = lambda: hk.nets.MLP(self.output_sizes+[D])
    if self.seperate_params:
      # [T, B, M, D]
      model_outputs = batch_multihead(
        x=model_input,
        fn=lambda: model_factory(),
        wrap_vmap=lambda fn: hk.BatchApply(fn),
        )
    else:
      # [T, B, M, D]
      model_outputs = hk.BatchApply(model_factory(), num_dims=3)(model_input)

    return {'model_outputs' : model_outputs}

class FarmCumulants(AuxilliaryTask):
  """docstring for FarmCumulants"""
  def __init__(self,
    module_cumulants=0,
    hidden_size=0,
    conv_size=0,
    layers=1,
    aggregation='sum',
    activation='none',
    construction='timestep',
    normalize_delta=True,
    normalize_cumulants=True,
    input_source='lstm',
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
    if hidden_size > 0 and layers > 0:
      layers = [hidden_size]*layers + [module_cumulants]
    else:
      layers = [module_cumulants]

    if module_cumulants > 0:
      self.cumulant_fn_factory = lambda: hk.nets.MLP(layers)

    self.module_cumulants = module_cumulants
    self.hidden_size = hidden_size
    aggregation = aggregation.lower()
    assert aggregation in ['sum', 'weighted', 'concat']
    self.aggregation = aggregation

    assert activation in ['identity', 'sigmoid']
    if activation == 'identity':
      activation = lambda x:x
    elif activation == 'sigmoid':
      activation = jax.nn.sigmoid
    self.activation = activation

    self.normalize_delta = normalize_delta
    self.normalize_cumulants = normalize_cumulants

    self.construction = construction.lower()
    self.construction_options = ['timestep', 'delta', 'concat']


    assert input_source in ['lstm', 'conv']
    self.input_source = input_source
    self.conv_size = conv_size

  def __call__(self, memory_out, predictions, **kwargs):
    if self.input_source == 'conv':
      inputs = memory_out.attn
    else:
      inputs = memory_out.hidden

    if hk.running_init():
      # during init, T=1
      inputs = jnp.concatenate((inputs, inputs), axis=0)

    assert self.construction in self.construction_options
    if self.construction == 'delta':
      states = inputs[:-1]  # [T, B, M, D]
      next_states = inputs[1:]  # [T, B, M, D]

      cumulants = next_states - states
      if self.normalize_delta:
        cumulants = cumulants / (1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    elif self.construction == 'concat':
      states = inputs[:-1]  # [T, B, M, D]
      next_states = inputs[1:]  # [T, B, M, D]
      cumulants = jnp.concatenate((next_states, states), axis=-1)

    elif self.construction == 'timestep':
      cumulants = inputs

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

    cumulants = self.activation(cumulants)

    return {'cumulants' : cumulants}

class FarmIndependentCumulants(FarmCumulants):
  """Each FARM module predicts its own set of cumulants"""

  def __init__(self, *args, seperate_params,
    normalize_state=False,
    relational_net=lambda x:x,
    **kwargs):
    super(FarmIndependentCumulants, self).__init__(*args, **kwargs)
    self.seperate_params = seperate_params
    self.construction_options = ['timestep', 'delta', 'concat', 'delta_concat']
    assert self.construction in self.construction_options
    self.normalize_state = normalize_state
    self.relational_net = relational_net

  def __call__(self, memory_out, predictions, **kwargs):
    if self.input_source == 'conv':
      # [T, B, N, H, W, D]
      # inputs = memory_out.attn
      # T, B, N = inputs.shape[:3]
      # D = inputs.shape[-1]
      # # compress so not so many params
      # if self.conv_size > 0:
      #   if self.seperate_params:
      #     inputs = batch_multihead(
      #       x=inputs,
      #       fn=lambda: hk.Conv2D(self.conv_size, [1, 1], 1),
      #       wrap_vmap=lambda fn: hk.BatchApply(fn),
      #       )
      #   else:
      #     inputs = hk.BatchApply(hk.Conv2D(self.conv_size, [1, 1], 1), num_dims=3)(inputs)
      # inputs = inputs.reshape(T, B, N, -1)
      raise RuntimeError("remove")

    else:
      inputs = memory_out.hidden

    if hk.running_init():
      # during init, T=1
      inputs = jnp.concatenate((inputs, inputs), axis=0)

    if self.construction == 'delta':
      states = inputs[:-1]  # [T, B, M, D]
      next_states = inputs[1:]  # [T, B, M, D]

      cumulants = next_states - states
      if self.normalize_delta:
        cumulants = cumulants / (1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))

    elif self.construction == 'delta_concat':
      states = inputs[:-1]  # [T, B, M, D]
      next_states = inputs[1:]  # [T, B, M, D]

      cumulants = next_states - states

      if self.normalize_delta:
        cumulants = cumulants / (1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
      if self.normalize_state:
        states = states / (1e-5+jnp.linalg.norm(states, axis=-1, keepdims=True))
      cumulants = jnp.concatenate((states, cumulants), axis=-1)

    elif self.construction == 'concat':
      states = inputs[:-1]  # [T, B, M, D]
      next_states = inputs[1:]  # [T, B, M, D]
      cumulants = jnp.concatenate((next_states, states), axis=-1)

    elif self.construction == 'timestep':
      cumulants = inputs
    else:
      raise RuntimeError

    cumulants = hk.BatchApply(self.relational_net)(cumulants)

    if self.seperate_params:
      cumulants = batch_multihead(
        x=cumulants,
        fn=lambda: self.cumulant_fn_factory(),
        wrap_vmap=lambda fn: hk.BatchApply(fn),
        )
      # [T, B, M, D] --> [T, B, M*D]
      cumulants = cumulants.reshape(*cumulants.shape[:2], -1)
    else:
      cumlant_fn = self.cumulant_fn_factory()
      cumulants = hk.BatchApply(cumlant_fn, num_dims=3)(cumulants)
      # [T, B, M, D] --> [T, B, M*D]
      cumulants = cumulants.reshape(*cumulants.shape[:2], -1)


    if self.normalize_cumulants:
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))

    cumulants = self.activation(cumulants)
    return {'cumulants' : cumulants}
