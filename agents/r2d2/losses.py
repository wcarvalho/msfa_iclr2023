"""R2d@ loss."""
import dataclasses
from typing import Tuple

from acme import types
from acme.agents.jax.dqn import learning_lib
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils

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


    # ======================================================
    # process data
    # ======================================================
    data = jax_utils.batch_to_sequence(batch.data)
    keys, probs, *_ = batch.info


    observations, actions, rewards, discounts, extra = (data.observation,
                                                        data.action,
                                                        data.reward,
                                                        data.discount,
                                                        data.extras)
    unused_sequence_length, batch_size = actions.shape

    # Get initial state for the LSTM, either from replay or simply use zeros.
    if self._store_lstm_state:
      import ipdb; ipdb.set_trace()
      # core_state = tree.map_structure(lambda x: x[0], extra['core_state'])
      core_state = jax.tree_map(lambda x: x[0], extra['core_state'])
    else:
      core_state = network.initial_state(batch_size)
      import ipdb; ipdb.set_trace()


    # Before training, optionally unroll the LSTM for a fixed warmup period.
    burn_in_obs = jax.tree_map(lambda x: x[:self._burn_in_length],
                                     observations)
    burn_in_fn = lambda o, s: network.unroll(o, s, self.burn_in_length)
    _, core_state = burn_in_fn(burn_in_obs, core_state)
    _, target_core_state = burn_in_fn(burn_in_obs, target_core_state)

    # Don't train on the warmup period.
    observations, actions, rewards, discounts, extra = jax.tree_map(
        lambda x: x[self._burn_in_length:],
        (observations, actions, rewards, discounts, extra))

    # ======================================================
    # same as gradient tape
    # ======================================================
    # Forward pass.
    # Unroll the online and target Q-networks on the sequences.
    q_values, _ = self._network.unroll(observations, core_state,
                                       self._sequence_length)
    target_q_values, _ = self._target_network.unroll(observations,
                                                     target_core_state,
                                                     self._sequence_length)

    q_tm1 = network.apply(params, transitions.observation)
    q_t_value = network.apply(target_params, transitions.next_observation)
    q_t_selector = network.apply(params, transitions.next_observation)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    # Compute double Q-learning n-step TD-error.
    batch_error = jax.vmap(rlax.double_q_learning)
    td_error = batch_error(q_tm1, transitions.action, r_t, d_t, q_t_value,
                           q_t_selector)
    batch_loss = rlax.huber_loss(td_error, self.huber_loss_parameter)

    # Importance weighting.
    sample_info = sample.info
    importance_weights = (1. / probs).astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)

    # Reweight.
    loss = jnp.mean(importance_weights * batch_loss)  # []
    reverb_update = learning_lib.ReverbUpdate(
        keys=keys, priorities=jnp.abs(td_error).astype(jnp.float64))
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return loss, extra
