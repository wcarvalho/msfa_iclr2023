"""
A few classes/functions to pass as aux_tasks that yield different types of cumulants
We also put the task_embed functions in this file
"""

from acme.jax.networks import base
import haiku as hk
import jax
import jax.numpy as jnp
from modules.basic_archs import AuxilliaryTask

class LinearTaskEmbed(hk.Module):
  def __init__(self, out_dim):
    super(LinearTaskEmbed, self).__init__()
    self.dim = out_dim
    self.embedder = hk.Linear(output_size=out_dim,with_bias=False,w_init=hk.initializers.TruncatedNormal()) #default is mean 0, std 1

  def __call__(self, x):
    return self.embedder(x)

  @property
  def out_dim(self):
    return self.dim
"""
concat conv features of s and s', then dense layer to get to desired dims
use same setup structure as Wilka's example below
"""
class CumulantsFromConvTask(AuxilliaryTask):
  def __init__(self, mlp_args, normalize = False, activation='none'):
    super(CumulantsFromConvTask, self).__init__(
      unroll_only=True, timeseries=True)
    self.cumulant_fn = hk.nets.MLP(mlp_args)
    #self.convnet = hk.Sequential([hk.AvgPool(2,2,'VALID'),hk.Flatten(2)])
    self.normalize = normalize

    assert activation in ['identity', 'sigmoid']
    if activation == 'identity':
      activation = lambda x:x
    elif activation == 'sigmoid':
      activation = jax.nn.sigmoid
    self.activation = activation
  def __call__(self, obs, **kwargs):
    #obs is the conv output
    #I'm going to guess its shape is [T+1, B, height, width, depth]
    states = obs[:-1]
    next_states = obs[1:]
    concatted = jnp.concatenate((states, next_states),axis=-1) #concat along feature dim
    #conved = self.convnet(concatted)
    cumulants = self.cumulant_fn(concatted)
    cumulants = self.activation(cumulants)

    if self.normalize:
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    return {"cumulants": cumulants}

"""EXAMPLE OF A CUMULANTS TASK"""
class CumulantsFromMemoryAuxTask(AuxilliaryTask):
  """docstring for Cumulants"""
  def __init__(self, *args, construction='timestep', normalize=False, activation='none', **kwargs):
    super(CumulantsFromMemoryAuxTask, self).__init__(
      unroll_only=True, timeseries=True)
    self.cumulant_fn = hk.nets.MLP(*args, **kwargs)
    self.normalize = normalize

    self.construction = construction.lower()
    assert self.construction in ['timestep', 'delta', 'concat']

    assert activation in ['identity', 'sigmoid']
    if activation == 'identity':
      activation = lambda x:x
    elif activation == 'sigmoid':
      activation = jax.nn.sigmoid
    self.activation = activation

  def __call__(self, memory_out, **kwargs):
    if self.construction == 'delta':
      states = memory_out[:-1]  # [T, B, N, D]
      next_states = memory_out[1:]  # [T, B, N, D]
      cumulants = next_states - states
    elif self.construction == 'concat':
      states = memory_out[:-1]  # [T, B, N, D]
      next_states = memory_out[1:]  # [T, B, N, D]
      cumulants = jnp.concatenate((next_states, states), axis=-1)
    else:
      cumulants = memory_out

    cumulants = self.cumulant_fn(cumulants)
    cumulants = self.activation(cumulants)

    if self.normalize:
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    return {'cumulants' : cumulants}
