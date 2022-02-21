import functools

import rlax
import jax
import jax.numpy as jnp
import haiku as hk
from agents.td_agent import losses
from utils import data as data_utils

from losses import nstep

# class ValueAuxLoss:
#   """docstring for ValueAuxLoss"""
#   def __init__(self, coeff: float, discount: float,
#     clip_rewards: bool = False,
#     bootstrap_n: int = 5,
#     tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR):
#     self.coeff = coeff
#     self.discount = discount
#     self.clip_rewards = clip_rewards
#     self.bootstrap_n = bootstrap_n
#     self.tx_pair = tx_pair

#   def __call__(self, data, online_preds, online_state, target_preds, target_state):

#     T, B = online_preds.value.shape[:2] # [T, B, 1]
#     online_v = online_preds.value.reshape(T, B)
#     target_v = target_preds.value.reshape(T, B)


#     # discounts [T, B]
#     discounts = (data.discount * self.discount).astype(online_v.dtype).reshape(T, B)


#     cumulants = online_preds.cumulants  # predicted  [T, B, D]
#     task = data.observation.observation.task  # ground-truth  [T, B, D]
#     reward_pred = jnp.sum(cumulants*task, -1)  # dot product  [T, B]
#     reward_pred = reward_pred.reshape(T, B)

#     batch_loss = jax.vmap(jax.vmap(rlax.td_learning))(
#       online_v[:-1],
#       # data.action[:-1],
#       reward_pred[:-1],
#       discounts[:-1],
#       target_v[1:],
#       ).mean()

#     return self.coeff*batch_loss, {'loss_value': batch_loss}


class QLearningAuxLoss(nstep.QLearning):
  def __init__(self, coeff, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.coeff = coeff

  def __call__(self, data, online_preds, online_state, target_preds, target_state):

    def compute_q(sf, w):
      return jnp.sum(sf*w, axis=-1)

    # all are [T, B, N, A, C]
    # output will be [T, B, A]
    import ipdb; ipdb.set_trace()
    online_q = compute_q(online_preds.sf, online_preds.w_embed)
    target_q = compute_q(target_preds.sf, target_preds.w_embed)


    # VMAP over N dimenison
    _, batch_loss, metrics = super().__call__(
      online_q=online_q,
      target_q=target_q,
      rewards=data.reward,
      discount=data.discount,
      actions=data.action)
    batch_loss = batch_loss.mean()

    metrics.update({'loss_qlearning': batch_loss})

    return self.coeff*batch_loss, metrics



