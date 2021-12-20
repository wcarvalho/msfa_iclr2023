import jax
import jax.numpy as jnp

import chex
Array = chex.Array


def n_step_bootstrapped_returns(
    c_t: Array,
    discount_t: Array,
    v_t: Array,
    n: int,
    lambda_t: float = 1.,
    stop_target_gradients: bool = False,
) -> Array:
  """Computes strided n-step bootstrapped return targets over a sequence.
  The returns are computed according to the below equation iterated `n` times:
     Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].
  When lambda_t == 1. (default), this reduces to
     Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).
  Args:
    c_t: cumulants at times [1, ..., T, D].
    discount_t: discounts at times [1, ..., T].
    v_t: state or state-action values to bootstrap from at time [1, ...., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    lambda_t: lambdas at times [1, ..., T]. Shape is [], or [T-1].
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
  Returns:
    estimated bootstrapped returns at times [0, ...., T-1]
  """
  # chex.assert_rank([c_t, discount_t, v_t, lambda_t], [1, 1, 1, {0, 1}])
  chex.assert_type([c_t, discount_t, v_t, lambda_t], float)
  # chex.assert_equal_shape([c_t, discount_t, v_t])
  seq_len = c_t.shape[0]
  # Maybe change scalar lambda to an array.
  lambda_t = jnp.ones_like(discount_t) * lambda_t

  # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
  pad_size = min(n - 1, seq_len)
  targets = jnp.concatenate([v_t[n - 1:], jnp.array([v_t[-1]] * pad_size)])

  # Pad sequences. Shape is now (T + n - 1,).
  cum_dims = c_t.shape[1:]
  c_t = jnp.concatenate([c_t, jnp.zeros((n - 1, *cum_dims))])
  dis_dims = discount_t.shape[1:]
  discount_t = jnp.concatenate([discount_t, jnp.ones((n - 1, *dis_dims))])
  lambda_t = jnp.concatenate([lambda_t, jnp.ones((n - 1, *dis_dims))])
  v_t = jnp.concatenate([v_t, jnp.array([v_t[-1]] * (n - 1))])

  # figure our how to expand {discount & lambda} for broadcasting
  rank_disc = len(discount_t.shape)
  rank_target = len(targets.shape)
  disc_expand = tuple(range(rank_disc, rank_target))

  # Work backwards to compute n-step returns.
  for i in reversed(range(n)):
    c_ = c_t[i:i + seq_len]
    discount_ = jnp.expand_dims(discount_t[i:i + seq_len], disc_expand)
    lambda_ = jnp.expand_dims(lambda_t[i:i + seq_len], disc_expand)
    v_ = v_t[i:i + seq_len]


    targets = c_ + discount_ * ((1. - lambda_) * v_ + lambda_ * targets)

  return jax.lax.select(stop_target_gradients,
                        jax.lax.stop_gradient(targets), targets)


def batched_index(
    values: Array, indices: Array, keepdims: bool = False,
) -> Array:
  """Index into 2nd to last dimension of a tensor, preserving all others dims.
  Args:
    values: a tensor of shape [..., D],
    indices: indices of shape [...].
    keepdims: whether to keep the final dimension.
    cumulant_dims: how many dimensions cumulant has. scalars=0. vectors=1. etc.
  Returns:
    a tensor of shape [...] or [..., 1].
  """
  indexed = jnp.take_along_axis(values, indices[..., None, None], axis=-2)
  if not keepdims:
    indexed = jnp.squeeze(indexed, axis=-2)

  return indexed

def n_step_td_learning(
    v_tm1: Array,
    a_tm1: Array,
    target_v_t: Array,
    a_t: Array,
    c_t: Array,
    discount_t: Array,
    n: int,
    stop_target_gradients: bool = True,
) -> Array:
  """Calculates transformed n-step TD errors.
  See "Recurrent Experience Replay in Distributed Reinforcement Learning" by
  Kapturowski et al. (https://openreview.net/pdf?id=r1lyTjAqYX).
  Args:
    v_tm1: values at times [0, ..., T - 1].
    a_tm1: action index at times [0, ..., T - 1].
    target_v_t: target values at time [1, ..., T].
    a_t: action index at times [[1, ... , T]] used to select target values to
      bootstrap from; max(target_v_t) for normal Q-learning, max(regular_v_t) 
      for double Q-learning.
    c_t: cumulant at times [1, ..., T].
    discount_t: discount at times [1, ..., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
    tx_pair: TxPair of value function transformation and its inverse.
  Returns:
    Transformed N-step TD error.
  """

  # below is just used to compute bootstrapped return
  v_t = batched_index(target_v_t, a_t)
  target_tm1 = n_step_bootstrapped_returns(
      c_t, discount_t, v_t, n,
      stop_target_gradients=stop_target_gradients)

  v_a_tm1 = batched_index(v_tm1, a_tm1)
  return target_tm1 - v_a_tm1