import functools

import rlax
import jax
import jax.numpy as jnp
import haiku as hk
from agents.td_agent import losses
from utils import data as data_utils



class QLearning:
  """docstring for ValueAuxLoss"""
  def __init__(self,
    discount: float,
    clip_rewards: bool = False,
    bootstrap_n: int = 5,
    tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR):
    self.discount = discount
    self.clip_rewards = clip_rewards
    self.bootstrap_n = bootstrap_n
    self.tx_pair = tx_pair

  def __call__(self, online_q : jnp.ndarray, target_q : jnp.ndarray, discount : jnp.ndarray, rewards : jnp.ndarray, actions : jnp.ndarray):
    """Summary
    
    Args:
        online_q (TYPE): T x B x A
        target_q (TYPE): T x B x A
        discount (TYPE): T x B
        rewards (TYPE): T x B
        actions (TYPE): T x B
    
    Returns:
        TYPE: Description
    """
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_q, axis=-1) # [T, B]
    # Preprocess discounts & rewards.
    discounts = (discount * self.discount).astype(online_q.dtype)
    if self.clip_rewards:
      rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)
    rewards = rewards.astype(online_q.dtype)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n,
            tx_pair=self.tx_pair),
        in_axes=1,
        out_axes=1)
    # TODO(b/183945808): when this bug is fixed, truncations of actions,
    # rewards, and discounts will no longer be necessary.
    return batch_td_error_fn(
        online_q[:-1],
        actions[:-1],
        target_q[1:],
        selector_actions[1:],
        rewards[:-1],
        discounts[:-1])

