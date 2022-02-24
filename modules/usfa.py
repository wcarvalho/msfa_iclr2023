from typing import Sequence, Optional
from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib

import functools
import haiku as hk
from utils import data as data_utils

from modules.embedding import OneHotTask
from modules.duelling import DuellingSfQNet

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
  ):
    super().__init__(name='sf_network')
    self.num_actions = num_actions
    self.num_cumulants = num_cumulants

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
    sf_input_fn = None,
    task_embed: int = 0,
    duelling: bool = False,
    normalize_task: bool = False,
    z_as_train_task: bool = False,
    ):
    super(UsfaHead, self).__init__()
    self.num_actions = num_actions
    self.state_dim = state_dim
    self.hidden_size = hidden_size
    self.var = variance
    self.nsamples = nsamples
    self.z_as_train_task = z_as_train_task


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
      self.sf_q_net = DuellingSfQNet(num_actions=num_actions, num_cumulants=self.sf_out_dim,
        hidden_sizes=[hidden_size])
    else:
      self.sf_q_net = SfQNet(num_actions=num_actions, num_cumulants=self.sf_out_dim,
        hidden_sizes=[hidden_size])

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

    w_train = jax.vmap(self.task_embed)(inputs.w_train) # [N, D]
    N = w_train.shape[0]
    if self.normalize_task:
      w_train = w_train/(1e-5+jnp.linalg.norm(w_train, axis=-1, keepdims=True))

    # -----------------------
    # policies + embeddings
    # -----------------------
    z = data_utils.expand_tile_dim(w_train, axis=0, size=B)
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
    sf_input = self.sf_input_fn(inputs.memory_out, z_embedding)

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

    w_expand = add_actions_dimension(w_expand)
    z = add_actions_dimension(z)

    # -----------------------
    # compute successor features
    # -----------------------
    # [B, N, A, S], [B, N, A]
    if z_as_task:
      sf, q_values = hk.BatchApply(self.sf_q_net)(sf_input, z)
    else:
      sf, q_values = hk.BatchApply(self.sf_q_net)(sf_input, w_expand)

    # [B, A]
    q_values = jnp.max(q_values, axis=1)

    return USFAPreds(
      sf=sf,
      z=z,
      q=q_values,
      w=w)


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

class UniqueStatePolicyPairs(StatePolicyCombination):
  def __call__(self, memory_out, z_embedding):
    return jax.vmap(data_utils.meshgrid)(memory_out, z_embedding)

  def augment_task(self, memory_out, w):
    repeat = functools.partial(
      jnp.repeat,
      repeats=memory_out.shape[1],
      axis=0)
    w = jax.vmap(repeat)(w)
    return w

class CumulantsAuxTask(hk.Module):
  """docstring for Cumulants"""
  def __init__(self, normalize=False, *args, **kwargs):
    super(Cumulants, self).__init__()
    self.cumulant_fn = hk.nets.MLP(*args, **kwargs)
    self.normalize = normalize

  def __call__(self, memory_out, **kwargs):
    cumulants = self.cumulant_fn(memory_out)
    if self.normalize:
      import ipdb; ipdb.set_trace()
      cumulants = cumulants/(1e-5+jnp.linalg.norm(cumulants, axis=-1, keepdims=True))
    return {'cumulants' : cumulants}
