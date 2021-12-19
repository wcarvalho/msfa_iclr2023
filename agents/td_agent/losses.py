"""R2D2 loss."""
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


from agents.td_agent.networks import TDNetworkFns
from agents.td_agent import td


@dataclasses.dataclass
class RecurrentTDLearning(learning_lib.LossFn):
  """R2D2 Learning."""
  discount: float = 0.99
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # More than DQN
  max_replay_size: int = 1_000_000
  store_lstm_state: bool = True
  max_priority_weight: float = 0.9
  bootstrap_n: int = 5
  importance_sampling_exponent: float = 0.2

  burn_in_length: int = None
  clip_rewards : bool = False
  max_abs_reward: float = 1.

  def error(self):
    raise NotImplementedError

  def __call__(
      self,
      network: TDNetworkFns,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key_grad: networks_lib.PRNGKey,
    ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""

    unroll = network.unroll  # convenienve

    # Convert sample data to sequence-major format [T, B, ...].
    data = jax_utils.batch_to_sequence(batch.data)

    # Get core state & warm it up on observations for a burn-in period.
    if self.store_lstm_state:
      # Replay core state.
      online_state = jax.tree_map(lambda x: x[0], data.extras['core_state'])
    else:
      _, batch_size = data.action.shape
      key_grad, key = jax.random.split(key_grad)
      online_state = network.initial_state.apply(params, key, batch_size)
    target_state = online_state

    # Maybe burn the core state in.
    burn_in_length = self.burn_in_length
    if burn_in_length:
      burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      _, online_state = unroll.apply(params, key1, burn_obs, online_state)
      _, target_state = unroll.apply(target_params, key2, burn_obs,
                                     target_state)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    # Unroll on sequences to get online and target Q-Values.
    key1, key2 = jax.random.split(key_grad)

    online_q, online_state = unroll.apply(params, key1, data.observation, online_state)
    target_q, target_state = unroll.apply(target_params, key2, data.observation,
                               target_state)

    batch_td_error, batch_loss = self.error(data, online_q, online_state, target_q, target_state)

    # Importance weighting.
    probs = batch.info.probability
    importance_weights = (1. / (probs + 1e-6)).astype(online_q.dtype)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    mean_loss = jnp.mean(importance_weights * batch_loss)

    # Calculate priorities as a mixture of max and mean sequence errors.
    abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = (max_priority + mean_priority)


    reverb_update = learning_lib.ReverbUpdate(
        keys=batch.info.key,
        priorities=priorities
        )
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return mean_loss, extra



def r2d2_loss_kwargs(config):
  return dict(
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      burn_in_length=config.burn_in_length,
      max_replay_size=config.max_replay_size,
      store_lstm_state=config.store_lstm_state,
      max_priority_weight=config.max_priority_weight,
      tx_pair=config.tx_pair,
      bootstrap_n=config.bootstrap_n,
      clip_rewards=config.clip_rewards,
  )

@dataclasses.dataclass
class R2D2Learning(RecurrentTDLearning):
  def error(self, data, online_q, online_state, target_q, target_state):
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_q, axis=-1)
    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype)
    rewards = data.reward
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
    batch_td_error = batch_td_error_fn(
        online_q[:-1],
        data.action[:-1],
        target_q[1:],
        selector_actions[1:],
        rewards[:-1],
        discounts[:-1])
    batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)
    return batch_td_error, batch_loss



def usfa_loss_kwargs(config):
  return dict(
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      burn_in_length=config.burn_in_length,
      max_replay_size=config.max_replay_size,
      store_lstm_state=config.store_lstm_state,
      max_priority_weight=config.max_priority_weight,
      tx_pair=config.tx_pair,
      bootstrap_n=config.bootstrap_n,
      clip_rewards=config.clip_rewards,
  )


@dataclasses.dataclass
class USFALearning(RecurrentTDLearning):
  def error(self, data, online_q, online_state, target_q, target_state):

    # all are [T, B, N, A, D], N = num policies, A = actions, D = dim state
    online_sf = online_state.sf 
    online_z = online_state.policy_zeds
    target_sf = target_state.sf
    target_z = target_state.policy_zeds
    npolicies = online_sf.shape[2]

    # Get value-selector actions from online Q-values for double Q-learning.
    # wil do average over [T, B, D]
    new_q =  (online_sf*online_z).sum(axis=-1) # overwrite, [T, B, N, A]
    target_actions = jnp.argmax(new_q, axis=-1) # [T, B, N]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(new_q.dtype)
    cumulants = data.observation.observation.state_features # [T, B, D]
    cumulants = jnp.expand_dims(cumulants, axis=2)
    cumulants = jnp.tile(cumulants, [1,1, npolicies, 1]) # [T, B, N, D]

    # actions used for online_sf
    online_actions = jnp.expand_dims(data.action, axis=2)
    online_actions = jnp.tile(online_actions, [1,1, npolicies]) # [T, B, N, D]
    # # Get N-step transformed TD error and loss.
    # batch_td_error_fn = jax.vmap(
    #     functools.partial(
    #         td.n_step_td_learning,
    #         n=self.bootstrap_n),
    #     in_axes=1, out_axes=1) # batch axis

    batch_td_error_fn = functools.partial(
            td.n_step_td_learning,
            n=self.bootstrap_n)

    # [T, B, N, D]
    batch_td_error = batch_td_error_fn(
        online_sf[:-1],
        online_actions[:-1],
        target_sf[1:],
        target_actions[1:],
        cumulants[:-1],
        discounts[:-1]) 


    batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=(0, 2, 3))
    return batch_td_error, batch_loss
