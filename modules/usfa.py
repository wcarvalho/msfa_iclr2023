from typing import Sequence, Optional
from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib

import functools
import haiku as hk
from utils import data as data_utils

from modules.basic_archs import AuxilliaryTask
from modules.embedding import OneHotTask
from modules.duelling import DuellingSfQNet
from utils import vmap
from utils import data as data_utils

class USFAPreds(NamedTuple):
  q: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  z: jnp.ndarray  # policy vector
  w: jnp.ndarray  # task vector (potentially embedded)

class USFAInputs(NamedTuple):
  w: jnp.ndarray  # task vector
  memory_out: jnp.ndarray  # memory output (e.g. LSTM)
  w_train: jnp.ndarray = None  # train task vectors


def sample_gauss(mean, var, key, nsamples, axis):
  # gaussian (mean=mean, var=.1I)
  # mean = [B, D]
  samples = data_utils.expand_tile_dim(mean, axis=axis, size=nsamples)
  dims = samples.shape # [B, N, D]
  samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
  return samples.astype(mean.dtype)


class SfQNet(hk.Module):
  """A MLP SF Q-network."""

  def __init__(
      self,
      num_actions: int,
      num_cumulants: int,
      hidden_sizes: Sequence[int],
      multihead: bool=False,
  ):
    super().__init__(name='sf_network')
    self.num_actions = num_actions
    self.num_cumulants = num_cumulants
    self.multihead = multihead
    if multihead:
      self.mlp_factory = lambda: hk.nets.MLP([
          *hidden_sizes, num_actions])
    else:
      self.mlp = hk.nets.MLP([
          *hidden_sizes, num_actions * num_cumulants])

  def __call__(self, inputs: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Forward pass of the duelling network.
    
    Args:
        inputs (jnp.ndarray): B x Z
        w (jnp.ndarray): B x A x C
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    if self.multihead:
      # [B, z] --> [B, C, z]
      inputs = data_utils.expand_tile_dim(inputs, size=self.num_cumulants, axis=1)
      # make C copies of MLP 
      sf = vmap.batch_multihead(
        fn=self.mlp_factory,
        x=inputs)
      # [B, A, C]
      sf = jnp.transpose(sf, (0, 2, 1))
    else:
      # [B, A * C]
      sf = self.mlp(inputs)
      # [B, A, C]
      sf = jnp.reshape(sf, [sf.shape[0], self.num_actions, self.num_cumulants])

    q_values = jnp.sum(sf*w, axis=-1) # [B, A]

    return sf, q_values


# ======================================================
# Architectures
# ======================================================
class UsfaHead(hk.Module):
  """Page 17."""
  def __init__(self,
    num_actions: int,
    state_dim: int,
    hidden_size : int=128,
    policy_size : int=32,
    policy_layers : int=2,
    variance: float=0.1,
    nsamples: int=30,
    sf_input_fn: hk.Module = None,
    task_embed: int = 0,
    duelling: bool = False,
    normalize_task: bool = False,
    z_as_train_task: bool = False,
    multihead: bool = False,
    concat_w: bool = False,
    ):
    """Summary
    
    Args:
        num_actions (int): Description
        state_dim (int): Description
        hidden_size (int, optional): hidden size of SF MLP network
        policy_size (int, optional): dimensionality of each layer of policy embedding network
        policy_layers (int, optional): layers for policy embedding net
        variance (float, optional): variances of sampling
        nsamples (int, optional): number of policies
        sf_input_fn (None, optional): module that combines lstm-state h with policy embedding z
        task_embed (int, optional): whether to embed task vector
        duelling (bool, optional): whether to use a duelling head
        normalize_task (bool, optional): whether to normalize task vector
        z_as_train_task (bool, optional): whether to dot-product with z-vector (True) or w-vector (False)
        multihead (bool, optional): whether to use seperate parameters for each cumulant
        concat_w (bool, optional): whether to have task w as input to SF (not just policy z)
    
    Raises:
        NotImplementedError: Description
    """
    super(UsfaHead, self).__init__()
    self.num_actions = num_actions
    self.state_dim = state_dim
    self.hidden_size = hidden_size
    self.var = variance
    self.nsamples = nsamples
    self.z_as_train_task = z_as_train_task
    self.multihead = multihead
    self.concat_w = concat_w


    # -----------------------
    # function to combine state + policy
    # -----------------------
    if sf_input_fn is None:
      sf_input_fn = ConcatFlatStatePolicy(hidden_size)
    self.sf_input_fn = sf_input_fn

    # -----------------------
    # function to embed task and figure out dim of SF
    # -----------------------
    if task_embed > 0:
      task_embed = OneHotTask(size=state_dim, dim=task_embed)
      self.sf_out_dim = task_embed.dim
    else:
      task_embed = lambda x:x
      self.sf_out_dim = state_dim

    self.task_embed = task_embed
    self.normalize_task = normalize_task

    # -----------------------
    # MLPs
    # -----------------------
    if policy_layers > 0:
      self.policynet = hk.nets.MLP(
          [policy_size]*policy_layers)
    else:
      self.policynet = lambda x:x

    if duelling:
      if multihead:
        raise NotImplementedError
      else:
        self.sf_q_net = DuellingSfQNet(num_actions=num_actions, 
          num_cumulants=self.sf_out_dim,
          hidden_sizes=[hidden_size])
    else:
      self.sf_q_net = SfQNet(num_actions=num_actions,
        num_cumulants=self.sf_out_dim,
        hidden_sizes=[hidden_size],
        multihead=multihead)

  def __call__(self,
    inputs : USFAInputs,
    key: networks_lib.PRNGKey) -> USFAPreds:

    # -----------------------
    # [potentially] embed task
    # -----------------------
    w = jax.vmap(self.task_embed)(inputs.w) # [B, D_w]
    if self.normalize_task:
      w = w/(1e-5+jnp.linalg.norm(w, axis=-1, keepdims=True))

    # -----------------------
    # policies + embeddings
    # -----------------------
    # sample N times: [B, D_w] --> [B, N, D_w]
    z_samples = sample_gauss(mean=w, var=self.var, key=key, nsamples=self.nsamples, axis=-2)

    # combine samples with original task vector
    z_base = jnp.expand_dims(w, axis=1) # [B, 1, D_w]
    z = jnp.concatenate((z_base, z_samples), axis=1)  # [B, N+1, D_w]
    return self.sfgpi(inputs=inputs, z=z, w=w,
      key=key,
      z_as_task=self.z_as_train_task)

  def evaluation(self,
    inputs : USFAInputs,
    key: networks_lib.PRNGKey) -> USFAPreds:

    # -----------------------
    # embed task
    # -----------------------
    w = jax.vmap(self.task_embed)(inputs.w) # [B, D]
    B = w.shape[0]
    if self.normalize_task:
      w = w/(1e-5+jnp.linalg.norm(w, axis=-1, keepdims=True))

    # train basis (z)
    w_train = jax.vmap(self.task_embed)(inputs.w_train) # [N, D]
    N = w_train.shape[0]
    if self.normalize_task:
      w_train = w_train/(1e-5+jnp.linalg.norm(w_train, axis=-1, keepdims=True))

    # -----------------------
    # policies + embeddings
    # -----------------------
    z = data_utils.expand_tile_dim(w_train, axis=0, size=B)
    # z = [B, N, D]
    # w = [B, D]
    preds = self.sfgpi(inputs=inputs, z=z, w=w, key=key)

    return preds

  def sfgpi(self,
    inputs: USFAInputs,
    z: jnp.ndarray,
    w: jnp.ndarray,
    key: networks_lib.PRNGKey,
    z_as_task: bool = False) -> USFAPreds:
    """Summary
    
    Args:
        inputs (USFAInputs): Description
        z (jnp.ndarray): B x N x D
        w (jnp.ndarray): B x D
        key (networks_lib.PRNGKey): Description
    
    Returns:
        USFAPreds: Description
    """

    z_embedding = hk.BatchApply(self.policynet)(z) # [B, N, D_z]
    sf_input = self.sf_input_fn(inputs.memory_out, z_embedding) # [B, N, D_s]
    # -----------------------
    # prepare task vectors
    # -----------------------
    # [B, D_z] --> [B, N, D_z]
    nz = z.shape[1]
    w_expand = data_utils.expand_tile_dim(w, axis=1, size=nz)

    # [B, N, D_w] --> [B, N, A, D_w]
    def add_actions_dimension(task):
      task = self.sf_input_fn.augment_task(memory_out=inputs.memory_out, w=task)
      task = data_utils.expand_tile_dim(task, axis=2, size=self.num_actions)
      return task

    z = add_actions_dimension(z)

    # -----------------------
    # compute successor features
    # -----------------------
    if self.concat_w:
      data_utils.expand_tile_dim(w, axis=2, size=self.num_actions)
      sf_input = jnp.concatenate((sf_input, w_expand), axis=-1)

    # inputs = [B, N, D_s], [B, N, A, D_w]
    # ouputs = [B, N, A, D_w], [B, N, A]
    if z_as_task:
      sf, q_values = hk.BatchApply(self.sf_q_net)(sf_input, z)
    else:
      w_expand = add_actions_dimension(w_expand)
      sf, q_values = hk.BatchApply(self.sf_q_net)(sf_input, w_expand)

    # -----------------------
    # GPI
    # -----------------------
    # [B, A]
    q_values = jnp.max(q_values, axis=1)

    return USFAPreds(
      sf=sf,       # [B, N, A, D_w]
      z=z,         # [B, N, A, D_w]
      q=q_values,  # [B, N, A]
      w=w)         # [B, D_w]


  @property
  def out_dim(self):
    return self.sf_out_dim

class StatePolicyCombination(hk.Module):
  def __call__(self, memory_out, z_embedding):
    raise NotImplementedError

  def augment_task(self, memory_out, w):
    return w

class ConcatFlatStatePolicy(StatePolicyCombination):
  """docstring for ConcatFlatStatePolicy"""
  def __init__(self, hidden_size):
    super(ConcatFlatStatePolicy, self).__init__()
    if hidden_size > 0:
      self.statefn = hk.nets.MLP(
          [hidden_size],
          activate_final=True)
    else:
      self.statefn = lambda x: x

  def __call__(self, memory_out, z_embedding):
    nsamples = z_embedding.shape[1]
    state = self.statefn(memory_out)
    # [B, S] --> # [B, N, S]
    state = data_utils.expand_tile_dim(state, size=nsamples, axis=-2)

    return jnp.concatenate((state, z_embedding), axis=-1)

class UniqueStatePolicyPairs(ConcatFlatStatePolicy):
  """For {z_1, ..., z_m}, {h_1, ..., h_n}, create m x n pairs:
  {(z_1, h_1), ..., (z_m, h_n)}
  """

  def __call__(self, memory_out, z_embedding):
    state = self.statefn(memory_out)
    return jax.vmap(data_utils.meshgrid)(state, z_embedding)

  def augment_task(self, memory_out, w):
    repeat = functools.partial(
      jnp.repeat,
      repeats=memory_out.shape[1],
      axis=0)
    w = jax.vmap(repeat)(w)
    return w

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
