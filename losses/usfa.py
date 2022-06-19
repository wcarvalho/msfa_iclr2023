import functools

import rlax
import jax
import jax.numpy as jnp
import haiku as hk
from agents.td_agent import losses
from utils import data as data_utils
import optax
from losses import nstep
from losses.utils import episode_mean, make_episode_mask

def compute_q(sf, w):
  return jnp.sum(sf*w, axis=-1)

class QLearningAuxLoss(nstep.QLearning):
  def __init__(self,
    coeff,
    *args,
    sched_end=None,
    sched_start_val=1.,
    sched_end_val=1e-4,
    add_bias=False,
    stop_w_grad=False,
    mask_loss=False,
    **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff
    self.sched_end = sched_end
    self.sched_start_val = sched_start_val
    self.sched_end_val = sched_end_val
    self.add_bias = add_bias
    self.stop_w_grad = stop_w_grad
    self.mask_loss = mask_loss
    if sched_end:
      self.schedule = optax.linear_schedule(
                  init_value=sched_start_val,
                  end_value=sched_end_val,
                  transition_steps=sched_end)

  def __call__(self, data, online_preds, target_preds, steps, **kwargs):

    online_w = online_preds.w  # [T, B, C]
    target_w = target_preds.w  # [T, B, C]
    online_sf = online_preds.sf[:,:,0]  # [T, B, A, C]
    target_sf = target_preds.sf[:,:,0]  # [T, B, A, C]
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over A

    if self.stop_w_grad:
      online_w = jax.lax.stop_gradient(online_w)
      target_w = jax.lax.stop_gradient(target_w)

    # output is [T, B, A]
    online_q = compute_q_jax(online_sf, online_w)
    target_q = compute_q_jax(target_sf, target_w)

    if self.add_bias:
      online_q = online_q + online_preds.qbias
      target_q = target_q + target_preds.qbias


    batch_td_error = super().__call__(
      online_q=online_q,  # [T, B, A]
      target_q=target_q,  # [T, B, A]
      discount=data.discount,  # [T, B]
      rewards=data.reward,  # [T, B]
      actions=data.action)  # [T, B]

    # output is [B]
    batch_loss = 0.5 * jnp.square(batch_td_error)
    if self.mask_loss:
      batch_loss = episode_mean(
        x=batch_loss,
        mask=make_episode_mask(data)[:-1])
      batch_loss = batch_loss.mean()
    else:
      batch_loss = batch_loss.mean()

    coeff = self.coeff
    if self.sched_end is not None and self.sched_end > 0:
      coeff = self.schedule(steps)*coeff

    loss = coeff*batch_loss

    metrics = {
      'loss_qlearning_sf_raw': batch_loss,
      'loss_qlearning_sf': loss,
      'z.q_sf_coeff': coeff,
      'z.q_sf_mean': online_q.mean(),
      'z.q_sf_var': online_q.var(),
      'z.q_sf_max': online_q.max(),
      'z.q_sf_min': online_q.min()}

    return loss, metrics


class QLearningEnsembleAuxLoss(QLearningAuxLoss):

  def __call__(self, data, online_preds, target_preds, steps, **kwargs):

    # [T, B, C]
    online_w = online_preds.w  # [T, B, C]
    target_w = target_preds.w  # [T, B, C]
    if self.stop_w_grad:
      online_w = jax.lax.stop_gradient(online_w)
      target_w = jax.lax.stop_gradient(target_w)


    # all data is [T, B, N, A, C]
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over N
    compute_q_jax = jax.vmap(compute_q_jax, in_axes=(3, None), out_axes=3)  # over A

    # output will be [T, B, N, A]
    online_q = compute_q_jax(online_preds.sf, online_w)
    target_q = compute_q_jax(target_preds.sf, target_w)

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
    if self.mask_loss:
      batch_loss = episode_mean(
        x=batch_loss,
        mask=make_episode_mask(data)[:-1])
      batch_loss = batch_loss.mean()
    else:
      batch_loss = batch_loss.mean()

    metrics = {
      'loss_qlearning_sf': batch_loss,
      'z.q_sf_mean': online_q.mean(),
      'z.q_sf_var': online_q.var(),
      'z.q_sf_max': online_q.max(),
      'z.q_sf_min': online_q.min()}

    loss = self.coeff*batch_loss
    return loss, metrics