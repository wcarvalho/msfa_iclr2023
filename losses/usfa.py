import functools

import rlax
import jax
import jax.numpy as jnp
import haiku as hk
from agents.td_agent import losses
from utils import data as data_utils

from losses import nstep

def compute_q(sf, w):
  return jnp.sum(sf*w, axis=-1)

class QLearningAuxLoss(nstep.QLearning):
  def __init__(self, coeff, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff

  def __call__(self, data, online_preds, target_preds, **kwargs):

    w = online_preds.w  # [T, B, C]
    online_sf = online_preds.sf[:,:,0]  # [T, B, A, C]
    target_sf = target_preds.sf[:,:,0]  # [T, B, A, C]
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over A

    # output is [T, B, A]
    online_q = compute_q_jax(online_sf, w)
    target_q = compute_q_jax(target_sf, w)

    batch_td_error = super().__call__(
      online_q=online_q,  # [T, B, N, A]
      target_q=target_q,  # [T, B, N, A]
      discount=data.discount,  # [T, B]
      rewards=data.reward,  # [T, B]
      actions=data.action)  # [T, B]

    batch_loss = 0.5 * jnp.square(batch_td_error).mean()

    metrics = {
      'loss_qlearning_sf': batch_loss,
      'z.q_sf_mean': online_q.mean(),
      'z.q_sf_var': online_q.var(),
      'z.q_sf_max': online_q.max(),
      'z.q_sf_min': online_q.min()}

    return self.coeff*batch_loss, metrics


class QLearningEnsembleAuxLoss(nstep.QLearning):
  def __init__(self, coeff, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff

  def __call__(self, data, online_preds, target_preds, **kwargs):

    # [T, B, C]
    w = online_preds.w

    # all data is [T, B, N, A, C]
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over N
    compute_q_jax = jax.vmap(compute_q_jax, in_axes=(3, None), out_axes=3)  # over A

    # output will be [T, B, N, A]
    online_q = compute_q_jax(online_preds.sf, w)
    target_q = compute_q_jax(target_preds.sf, w)

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
      'loss_qlearning_sf': batch_loss,
      'z.q_sf_mean': online_q.mean(),
      'z.q_sf_var': online_q.var(),
      'z.q_sf_max': online_q.max(),
      'z.q_sf_min': online_q.min()}

    return self.coeff*batch_loss, metrics

