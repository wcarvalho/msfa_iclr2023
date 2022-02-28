from typing import NamedTuple

from acme.jax.networks import duelling
from acme.jax import networks as networks_lib

import jax
import jax.numpy as jnp

import haiku as hk

from utils import data as data_utils

from losses import nstep

class QEnsembleInputs(NamedTuple):
  w: jnp.ndarray  # task vector
  memory_out: jnp.ndarray  # memory output (e.g. LSTM)

class QEnsemblePreds(NamedTuple):
  q: jnp.ndarray  # chosen q-value
  all_q: jnp.ndarray # all q-values

def sample_gauss(mean, var, key, nsamples, axis):
  # gaussian (mean=mean, var=.1I)
  # mean = [B, D]
  samples = data_utils.expand_tile_dim(mean, axis=axis, size=nsamples)
  dims = samples.shape # [B, N, D]
  samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
  return samples.astype(mean.dtype)

class ConcatFlatStatePolicy(hk.Module):
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

class QEnsembleHead(hk.Module):
  """"""
  def __init__(self,
    num_actions: int,
    hidden_size : int=128,
    policy_size : int=32,
    policy_layers : int=2,
    variance: float=0.1,
    nsamples: int=30,
    q_input_fn = None,
    ):
    super(QEnsembleHead, self).__init__()
    self.num_actions = num_actions
    self.hidden_size = hidden_size
    self.var = variance
    self.nsamples = nsamples


    # -----------------------
    # function to combine state + policy
    # -----------------------
    if q_input_fn is None:
      q_input_fn = ConcatFlatStatePolicy(hidden_size)
    self.q_input_fn = q_input_fn

    # -----------------------
    # MLPs
    # -----------------------
    if policy_layers > 0:
      self.policynet = hk.nets.MLP(
          [policy_size]*policy_layers)
    else:
      self.policynet = lambda x:x

    self.q_net = duelling.DuellingMLP(
      num_actions=num_actions,
      hidden_sizes=[hidden_size])

  def __call__(self,
    inputs : QEnsembleInputs,
    key: networks_lib.PRNGKey) -> QEnsemblePreds:

    # -----------------------
    # [potentially] embed task
    # -----------------------
    w = inputs.w # [B, D_w]

    # -----------------------
    # policies + embeddings
    # -----------------------
    # sample N times: [B, D_w] --> [B, N, D_w]
    w_noise = sample_gauss(mean=w, var=self.var, key=key, nsamples=self.nsamples, axis=-2)

    # combine samples with original task vector
    w_base = jnp.expand_dims(w, axis=1) # [B, 1, D_w]
    w_all = jnp.concatenate((w_base, w_noise), axis=1)  # [B, N+1, D_w]


    w_embedding = hk.BatchApply(self.policynet)(w_all) # [B, N, D_z]
    q_input = self.q_input_fn(inputs.memory_out, w_embedding)

    # -----------------------
    # compute Q values and do GPI
    # -----------------------
    # [B, N, A, S], [B, N, A]
    all_q_values = hk.BatchApply(self.q_net)(q_input)

    # [B, A]
    q_values = jnp.max(all_q_values, axis=1)

    return QEnsemblePreds(
      all_q=all_q_values,
      q=q_values)

class QLearningEnsembleLoss(nstep.QLearning):
  def __init__(self, coeff, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff

  def __call__(self, data, online_preds, target_preds, **kwargs):

    # output will be [T, B, N, A]
    online_q = online_preds.all_q
    target_q = target_preds.all_q

    # VMAP over dimension = N
    q_learning = jax.vmap(
      super().__call__,
      in_axes=(2, 2, None, None, None))

    batch_td_error = q_learning(
      online_q,  # [T, B, N, A]
      target_q,  # [T, B, N, A]
      data.discount,  # [T, B]
      data.reward,  # [T, B]
      data.action)  # [T, B]

    batch_loss = 0.5 * jnp.square(batch_td_error).mean()

    metrics = {
      'loss_qlearning_ensemb': batch_loss,
      'z.q_ensemb_mean': online_q.mean(),
      'z.q_ensemb_var': online_q.var(),
      'z.q_ensemb_max': online_q.max(),
      'z.q_ensemb_min': online_q.min()}

    return self.coeff*batch_loss, metrics

