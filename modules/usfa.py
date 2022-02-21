from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib

import functools
import haiku as hk
from utils import data as data_utils

from modules.embedding import OneHotTask

class USFAPreds(NamedTuple):
  q: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  z: jnp.ndarray  # policy vector
  w_embed: jnp.ndarray  # embedding of task

class USFAInputs(NamedTuple):
  w: jnp.ndarray  # task vector
  memory_out: jnp.ndarray  # memory output (e.g. LSTM)


def sample_gauss(mean, var, key, nsamples, axis):
  # gaussian (mean=mean, var=.1I)
  # mean = [B, D]
  samples = data_utils.expand_tile_dim(mean, axis=axis, size=nsamples)
  dims = samples.shape # [B, N, D]
  samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
  return samples.astype(mean.dtype)


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
    variance: float=0.1,
    nsamples: int=30,
    sf_input_fn = None,
    task_embed: int = 0,
    normalize_task: bool = False,
    ):
    super(UsfaHead, self).__init__()
    self.num_actions = num_actions
    self.state_dim = state_dim
    self.hidden_size = hidden_size
    self.var = variance
    self.nsamples = nsamples
    self.ntask_vectors = nsamples + 1 # original + all samples

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
    self.policynet = hk.nets.MLP(
        [policy_size, policy_size])

    self.successorfn = hk.nets.MLP([
        policy_size + hidden_size,
        num_actions * self.sf_out_dim])



  def __call__(self,
    inputs : USFAInputs,
    key: networks_lib.PRNGKey) -> USFAPreds:

    # -----------------------
    # embed task
    # -----------------------
    w_embed = jax.vmap(self.task_embed)(inputs.w) # [B, D]
    if self.normalize_task:
      w_embed/(1e-5+jnp.linalg.norm(w_embed, axis=-1, keepdims=True))

    # -----------------------
    # policies + embeddings
    # -----------------------
    # sample N times: [B, S] --> [B, N, S]
    z_samples = sample_gauss(mean=w_embed, var=self.var, key=key, nsamples=self.nsamples, axis=-2)

    # combine samples with original task vector
    z_base = jnp.expand_dims(w_embed, axis=1)
    z = jnp.concatenate((z_base, z_samples), axis=1)  # [B, N+1, S]
    z_embedding = hk.BatchApply(self.policynet)(z) # [B, N, D]
    nz = z.shape[1]

    # -----------------------
    # compute successor features
    # -----------------------
    sf_input = self.sf_input_fn(inputs.memory_out, z_embedding)
    sf = hk.BatchApply(self.successorfn)(sf_input) # [B, N, A*S]

    # [B, N, A, S]
    sf = jnp.reshape(sf, [*sf.shape[:-1], self.num_actions, self.sf_out_dim])

    # -----------------------
    # Compute Q values --> Generalized Policy Improvement (best policy)
    # use task vector
    # -----------------------
    # [B, D] --> [B, N+1, D]
    w_embed_expand = data_utils.expand_tile_dim(w_embed, axis=-2, size=nz)

    def add_actions_dimension(task):
      task = self.sf_input_fn.augment_task(memory_out=inputs.memory_out, w=task)
      # [B, N, D] --> [B, N+1, A, D]
      task = data_utils.expand_tile_dim(task, axis=-2, size=self.num_actions)
      return task

    w_embed_expand = add_actions_dimension(w_embed_expand)
    z = add_actions_dimension(z)

    q_values = jnp.sum(sf*w_embed_expand, axis=-1) # [B, N, A]
    q_values = jnp.max(q_values, axis=-2) # [B, A]


    return USFAPreds(
      sf=sf,
      z=z,
      q=q_values,
      w_embed=w_embed)

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
    self.statefn = hk.nets.MLP(
        [hidden_size],
        activate_final=True)

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
