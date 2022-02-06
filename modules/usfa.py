from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib

import haiku as hk
from utils import data as data_utils

class USFAPreds(NamedTuple):
  q: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  z: jnp.ndarray  # policy vector

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


class UsfaHead(hk.Module):
  """Page 17."""
  def __init__(self,
    num_actions: int,
    state_dim: int,
    hidden_size : int=128,
    policy_size : int=32,
    variance: float=0.1,
    nsamples: int=30,
    ):
    super(UsfaHead, self).__init__()
    self.num_actions = num_actions
    self.state_dim = state_dim
    self.var = variance
    self.nsamples = nsamples
    self.statefn = hk.nets.MLP(
        [hidden_size],
        activate_final=True)

    self.policynet = hk.nets.MLP(
        [policy_size, policy_size])

    self.successorfn = hk.nets.MLP([
        policy_size + hidden_size,
        num_actions * state_dim])

  def __call__(self,
    inputs : USFAInputs,
    key: networks_lib.PRNGKey) -> USFAPreds:

    # -----------------------
    # policies + embeddings
    # -----------------------
    # [B, S] --> [B, N, S]
    z = sample_gauss(inputs.w, self.var, key, self.nsamples, axis=-2)
    z_embedding = hk.BatchApply(self.policynet)(z) # [B, N, D]

    # -----------------------
    # state
    # -----------------------
    # [B, S]
    state = self.statefn(inputs.memory_out)
    # [B, S] --> # [B, N, S]
    state = data_utils.expand_tile_dim(state, size=self.nsamples, axis=-2)

    # -----------------------
    # compute successor features
    # -----------------------
    sf_input = jnp.concatenate((state, z_embedding), axis=-1)
    sf = hk.BatchApply(self.successorfn)(sf_input) # [B, N, A*S]

    # [B, N, A, S]
    sf = jnp.reshape(sf, [*sf.shape[:-1], self.num_actions, self.state_dim])

    # -----------------------
    # Compute Q values --> Generalized Policy Improvement (best policy)
    # -----------------------
    # [B, N, S] --> [B, N, A, S]
    z = data_utils.expand_tile_dim(z, axis=-2, size=self.num_actions)
    q_values = jnp.sum(sf*z, axis=-1) # [B, N, A]
    q_values = jnp.max(q_values, axis=-2) # [B, A]

    return USFAPreds(
      sf=sf,
      z=z,
      q=q_values)


class RewardAuxTask(hk.Module):
  """docstring for RewardAuxTask"""
  def __init__(self, hidden_size, state_dim):
    super(RewardAuxTask, self).__init__()
    self.cumulant_fn = hk.nets.MLP(
        [hidden_size, state_dim],
        activate_final=False)

  def __call__(self, memory_out, **kwargs):
    return {'cumulants' : self.cumulant_fn(memory_out)}
