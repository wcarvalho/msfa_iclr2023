import functools

import rlax
import jax
import jax.numpy as jnp
import haiku as hk
from agents.td_agent import losses
from utils import data as data_utils

from losses import nstep
from losses.utils import episode_mean

def compute_q(sf, w):
  return jnp.sum(sf*w, axis=-1)

class QLearningAuxLoss(nstep.QLearning):
  def __init__(self,
    coeff,
    *args,
    sched_end=None,
    sched_start_val=1.,
    sched_end_val=0.,
    **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff
    self.sched_end = sched_end
    self.sched_start_val = sched_start_val
    self.sched_end_val = sched_end_val

  def __call__(self, data, online_preds, target_preds, steps, **kwargs):

    w = online_preds.w  # [T, B, C]
    online_sf = online_preds.sf[:,:,0]  # [T, B, A, C]
    target_sf = target_preds.sf[:,:,0]  # [T, B, A, C]
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over A

    # output is [T, B, A]
    online_q = compute_q_jax(online_sf, w)
    target_q = compute_q_jax(target_sf, w)

    batch_td_error = super().__call__(
      online_q=online_q,  # [T, B, A]
      target_q=target_q,  # [T, B, A]
      discount=data.discount,  # [T, B]
      rewards=data.reward,  # [T, B]
      actions=data.action)  # [T, B]

    # output is [B]
    batch_loss = 0.5 * jnp.square(batch_td_error)
    batch_loss = episode_mean(
      x=batch_loss,
      done=data.discount[:-1])
    batch_loss = batch_loss.mean()


    metrics = {
      'loss_qlearning_sf': batch_loss,
      'z.q_sf_mean': online_q.mean(),
      'z.q_sf_var': online_q.var(),
      'z.q_sf_max': online_q.max(),
      'z.q_sf_min': online_q.min()}

    loss = self.coeff*batch_loss
    return loss, metrics


class QLearningEnsembleAuxLoss(QLearningAuxLoss):

  def __call__(self, data, online_preds, target_preds, steps, **kwargs):

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
      super(QLearningAuxLoss, self).__call__,
      in_axes=(2, 2, None, None, None))

    batch_td_error = q_learning(
      online_q,  # [T, B, N, A]
      target_q,  # [T, B, N, A]
      data.discount,  # [T, B]
      data.reward,  # [T, B]
      data.action)  # [T, B]

    # output is [B]
    batch_loss = 0.5 * jnp.square(batch_td_error).mean(2)
    batch_loss = episode_mean(
      x=batch_loss,
      done=data.discount[:-1])
    batch_loss = batch_loss.mean()

    metrics = {
      'loss_qlearning_sf': batch_loss,
      'z.q_sf_mean': online_q.mean(),
      'z.q_sf_var': online_q.var(),
      'z.q_sf_max': online_q.max(),
      'z.q_sf_min': online_q.min()}

    loss = self.coeff*batch_loss
    return loss, metrics