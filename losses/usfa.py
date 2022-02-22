import functools

import rlax
import jax
import jax.numpy as jnp
import haiku as hk
from agents.td_agent import losses
from utils import data as data_utils

from losses import nstep

class QLearningAuxLoss(nstep.QLearning):
  def __init__(self, coeff, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff

  def __call__(self, data, online_preds, target_preds, **kwargs):

    def compute_q(sf, w):
      return jnp.sum(sf*w, axis=-1)

    # [T, B, C]
    w_embed = online_preds.w_embed

    # all data is [T, B, N, A, C]
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over N
    compute_q_jax = jax.vmap(compute_q_jax, in_axes=(3, None), out_axes=3)  # over A

    # output will be [T, B, N, A]
    online_q = compute_q_jax(online_preds.sf, w_embed)
    target_q = compute_q_jax(target_preds.sf, w_embed)

    # VMAP over dimension = N
    q_learning = jax.vmap(
      super().__call__,
      in_axes=(2, 2, None, None, None))

    batch_td_error = q_learning(
      online_q,  # [T, B, N, A]
      target_q,  # [T, B, N, A]
      data.reward,  # [T, B]
      data.discount,  # [T, B]
      data.action)  # [T, B]

    batch_loss = 0.5 * jnp.square(batch_td_error).mean()

    metrics = {
      'loss_qlearning': batch_loss,
      'z.q_mean': online_q.mean(),
      'z.q_var': online_q.var(),
      'z.q_max': online_q.max(),
      'z.q_min': online_q.min()}

    return self.coeff*batch_loss, metrics



