"""R2d@ loss."""
import dataclasses
from typing import Tuple

from acme import types
from acme.agents.jax.dqn import learning_lib
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils

import functools
import haiku as hk
import jax
import jax.numpy as jnp
import reverb
import rlax
import tree


from agents.r2d2.networks import R2D2Network

@dataclasses.dataclass
class R2D2Learning(learning_lib.LossFn):
  """R2D2 Learning."""
  discount: float = 0.99
  huber_loss_parameter: float = 1. # TODO: check

  # More than DQN
  max_replay_size: int = 1_000_000
  store_lstm_state: bool = True
  max_priority_weight: float = 0.9
  n_step: int = 5
  importance_sampling_exponent: float = 0.2

  burn_in_length: int = None
  sequence_length: int = None


  def __call__(
      self,
      network: R2D2Network,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    import ipdb; ipdb.set_trace()
    # ======================================================
    # load data
    # ======================================================
    data = jax_utils.batch_to_sequence(batch.data)

    observations, actions, rewards, discounts, extra = (data.observation,
                                                        data.action,
                                                        data.reward,
                                                        data.discount,
                                                        data.extras)
    unused_sequence_length, batch_size = actions.shape

    # Get initial state for the LSTM, either from replay or simply use zeros.
    if self.store_lstm_state:
      import ipdb; ipdb.set_trace()
      core_state = jax.tree_map(lambda x: x[0], extra['core_state'])
    else:
      core_state = network.initial_state(params, batch_size)
    target_core_state = core_state

    # ======================================================
    # Apply Networks
    # ======================================================
    # Before training, optionally unroll the LSTM for a fixed warmup period.
    burn_in_obs = jax.tree_map(lambda x: x[:self.burn_in_length],
                                     observations)
    _, core_state = network.unroll(params, burn_in_obs, core_state)
    _, target_core_state = network.unroll(target_params, burn_in_obs, target_core_state)

    # Don't train on the warmup period.
    observations, actions, rewards, discounts, extra = jax.tree_map(
        lambda x: x[self.burn_in_length:],
        (observations, actions, rewards, discounts, extra))


    # Forward pass.
    # Unroll the online and target Q-networks on the sequences.
    q_values, _ = network.unroll(params, observations, core_state)
    target_q_values, _ = network.unroll(target_params, observations, target_core_state)


    # ======================================================
    # Prepare and align time-series data
    # ======================================================
    rewards = jax.tree_map(lambda x: x[:-1], rewards)
    discounts = jax.tree_map(lambda x: x[:-1], discounts)
    a_tm1 = jax.tree_map(lambda x: x[:-1], actions)
    a_t = jax.tree_map(lambda x: x[1:], actions)
    q_tm1 = jax.tree_map(lambda x: x[:-1], q_values)
    target_q_t=jax.tree_map(lambda x: x[1:], target_q_values)


    # ======================================================
    # Compute the transformed n-step loss.
    # ======================================================
    batch_error = jax.vmap(functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.n_step,
            tx_pair=rlax.SIGNED_HYPERBOLIC_PAIR))

    td_error = batch_error(
        q_tm1=q_tm1,
        a_tm1=a_tm1,
        target_q_t=target_q_t,
        a_t=a_t,
        r_t=rewards,
        discount_t=discounts,
    )


    # Sum over time dimension.
    batch_loss = 0.5 * jnp.sum(jnp.square(td_error), axis=0)


    # Calculate importance weights and use them to scale the loss.
    keys, probs, *_ = batch.info
    importance_weights = 1. / (self.max_replay_size * probs)  # [T, B]
    importance_weights = importance_weights.astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    # Reweight.
    loss = jnp.mean(batch_loss*importance_weights)  # []

    import ipdb; ipdb.set_trace()
    reverb_update = learning_lib.ReverbUpdate(
        keys=keys, priorities=jnp.abs(td_error).astype(jnp.float64))
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    import ipdb; ipdb.set_trace()
    return loss, extra
