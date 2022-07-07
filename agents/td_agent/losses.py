"""R2D2 loss."""
import dataclasses
from typing import Tuple, Sequence, List, Union, Callable

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


from agents.td_agent.types import TDNetworkFns, Predictions
from losses.utils import episode_mean, make_episode_mask

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
  loss_coeff: float = 1.
  mask_loss: bool = False

  # auxilliary tasks
  aux_tasks: Union[Callable, Sequence[Callable]]=None

  def error(self,
      data,
      online_preds : Predictions,
      online_state,
      target_preds : Predictions,
      target_state,
      **kwargs):
    """Summary
    
    Args:
        data (TYPE): Description
        online_preds (Predictions): Description
        online_state (TYPE): Description
        target_preds (Predictions): Description
        target_state (TYPE): Description

    Raises:
        NotImplementedError: Description
    """
    raise NotImplementedError

  def __call__(
      self,
      network: TDNetworkFns,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key_grad: networks_lib.PRNGKey,
      steps: int=None,
    ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""

    unroll = network.unroll  # convenience

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
      _, online_state = unroll.apply(params, key1, burn_obs, online_state, key2)
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      _, target_state = unroll.apply(target_params, key1, burn_obs,
                                     target_state, key2)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    # Unroll on sequences to get online and target Q-Values.

    key_grad, key1, key2 = jax.random.split(key_grad, 3)
    online_preds, online_state = unroll.apply(params, key1, data.observation, online_state, key2)
    key_grad, key1, key2 = jax.random.split(key_grad, 3)
    target_preds, target_state = unroll.apply(target_params, key1, data.observation,
                               target_state, key2)

    # -----------------------
    # main loss
    # -----------------------
    batch_td_error, batch_loss, metrics = self.error(
      data=data,
      online_preds=online_preds,
      online_state=online_state,
      target_preds=target_preds,
      target_state=target_state,
      steps=steps)

    # Importance weighting.
    probs = batch.info.probability
    importance_weights = (1. / (probs + 1e-6)).astype(online_preds.q.dtype)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    mean_loss = jnp.mean(importance_weights * batch_loss)

    Cls = lambda x: x.__class__.__name__
    metrics={
      Cls(self) : {
        **metrics,
        'loss_main':mean_loss,
        'z.importance': importance_weights.mean(),
        'z.reward' :data.reward.mean()
        }
      }
    mean_loss = self.loss_coeff*mean_loss

    # Calculate priorities as a mixture of max and mean sequence errors.
    abs_td_error = jnp.abs(batch_td_error).astype(online_preds.q.dtype)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = (max_priority + mean_priority)


    # -----------------------
    # auxilliary tasks
    # -----------------------
    if self.aux_tasks:
      aux_tasks = self.aux_tasks
      if not isinstance(aux_tasks, list): aux_tasks = [aux_tasks]

      for aux_task in aux_tasks:
        # does this aux task need a random key?
        kwargs=dict()
        if hasattr(aux_task, 'random') and aux_task.random:
          key_grad, key = jax.random.split(key_grad, 2)
          kwargs['key'] = key

        aux_loss, aux_metrics = aux_task(
          data=data,
          online_preds=online_preds,
          online_state=online_state,
          target_preds=target_preds,
          target_state=target_state,
          steps=steps,
          **kwargs)

        
        # metrics={**metrics,**aux_metrics}
        metrics[Cls(aux_task)] = aux_metrics
        mean_loss = mean_loss + aux_loss

      metrics[Cls(self)]['loss_w_aux'] = mean_loss

    reverb_update = learning_lib.ReverbUpdate(
        keys=batch.info.key,
        priorities=priorities
        )
    extra = learning_lib.LossExtra(metrics=metrics, reverb_update=reverb_update)
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
      loss_coeff=config.loss_coeff,
  )

@dataclasses.dataclass
class R2D2Learning(RecurrentTDLearning):

  loss: str = 'transformed_n_step_q_learning'

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    """R2D2 learning
    """
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_preds.q, axis=-1) # [T+1, B]
    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_preds.q.dtype)
    rewards = data.reward
    if self.clip_rewards:
      rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)
    rewards = rewards.astype(online_preds.q.dtype)

    # Get N-step transformed TD error and loss.
    if self.loss == "transformed_n_step_q_learning":
      tx_pair = rlax.SIGNED_HYPERBOLIC_PAIR
    elif self.loss == "n_step_q_learning":
      tx_pair = rlax.IDENTITY_PAIR
    else:
      raise NotImplementedError(self.loss)

    batch_td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n,
            tx_pair=tx_pair),
        in_axes=1,
        out_axes=1)
    # TODO(b/183945808): when this bug is fixed, truncations of actions,
    # rewards, and discounts will no longer be necessary.
    batch_td_error = batch_td_error_fn(
        online_preds.q[:-1], # [T+1] --> [T]
        data.action[:-1],    # [T+1] --> [T]
        target_preds.q[1:],  # [T+1] --> [T]
        selector_actions[1:],# [T+1] --> [T]
        rewards[:-1],        # [T+1] --> [T]
        discounts[:-1])      # [T+1] --> [T]

    # average over {T} --> # [B]
    if self.mask_loss:
      # [T, B]
      episode_mask = make_episode_mask(data, include_final=False)
      batch_loss = episode_mean(
        x=(0.5 * jnp.square(batch_td_error)),
        mask=episode_mask[:-1])
    else:
      batch_loss = 0.5 * jnp.square(batch_td_error).mean(axis=0)

    metrics = {
      'z.q_mean': online_preds.q.mean(),
      'z.q_var': online_preds.q.var(),
      'z.q_max': online_preds.q.max(),
      'z.q_min': online_preds.q.min()}

    return batch_td_error, batch_loss, metrics # [T-1, B], [B]


def cumulants_from_env(data, online_preds, online_state, target_preds, target_state):
  return data.observation.observation.state_features # [T, B, C]

def cumulants_from_preds(
  data,
  online_preds,
  online_state,
  target_preds,
  target_state,
  stop_grad=True,
  use_target=False):
  
  if use_target:
    cumulants = target_preds.cumulants
  else:
    cumulants = online_preds.cumulants
  if stop_grad:
    return jax.lax.stop_gradient(cumulants) # [T, B, C]
  else:
    return cumulants # [T, B, C]

def dummy_cumulants_from_env(data, online_preds, online_state, target_preds, target_state):
  state_features = data.observation.observation.state_features # [T, B, C]
  return jnp.zeros(state_features.shape)


@dataclasses.dataclass
class USFALearning(RecurrentTDLearning):

  extract_cumulants: Callable = cumulants_from_env
  shorten_data_for_cumulant: bool = False
  loss: str = 'n_step_q_learning'
  lambda_: float  = .9

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    assert self.loss in ['transformed_n_step_q_learning', 'transformed_q_lambda', 'q_lambda', 'n_step_q_learning'], "loss not recognized"

    # ======================================================
    # Prepare Data
    # ======================================================
    # all are [T+1, B, N, A, C]
    # N = num policies, A = actions, C = cumulant dim
    online_sf = online_preds.sf
    online_z = online_preds.z
    target_sf = target_preds.sf

    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(
      data=data, online_preds=online_preds, online_state=online_state,
      target_preds=target_preds, target_state=target_state)
    cumulants = cumulants.astype(online_sf.dtype)

    # Get selector actions from online Q-values for double Q-learning.
    online_q =  (online_sf*online_z).sum(axis=-1) # [T+1, B, N, A]
    selector_actions = jnp.argmax(online_q, axis=-1) # [T+1, B, N]
    online_actions = data.action # [T, B]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype) # [T, B]

    cumulants_T = cumulants.shape[0]
    data_T = online_sf.shape[0]

    if cumulants_T == data_T:
      # shorten cumulants
      cum_idx = data_T - 1
    elif cumulants_T == data_T - 1:
      # no need to shorten cumulants
      cum_idx = cumulants_T
    elif cumulants_T > data_T:
      raise RuntimeError("This should never happen?")
    else:
      raise NotImplementedError



    # ======================================================
    # Loss for SF
    # ======================================================
    def sf_loss(online_sf, online_actions, target_sf, selector_actions, cumulants, discounts):
      """Vmap over cumulant dimension.
      
      Args:
          online_sf (TYPE): [T, A, C]
          online_actions (TYPE): [T]
          target_sf (TYPE): [T, A, C]
          selector_actions (TYPE): [T]
          cumulants (TYPE): [T, C]
          discounts (TYPE): [T]

      Returns:
          TYPE: Description
      """
      # copies selector_actions, online_actions, vmaps over cumulant dim

      # go over cumulant axis
      if self.loss == "transformed_n_step_q_learning":
        td_error_fn = jax.vmap(
          functools.partial(
              rlax.transformed_n_step_q_learning,
              n=self.bootstrap_n,
              tx_pair=self.tx_pair),
          in_axes=(2, None, 2, None, 1, None), out_axes=1)

        td_error = td_error_fn(
          online_sf[:-1],       # [T, A, C] (vmap 2) 
          online_actions[:-1],  # [T]       (vmap None) 
          target_sf[1:],        # [T, A, C] (vmap 2) 
          selector_actions[1:], # [T]       (vmap None) 
          cumulants[:cum_idx],       # [T, C]    (vmap 1) 
          discounts[:-1])       # [T]       (vmap None)

      elif self.loss == "n_step_q_learning":
        td_error_fn = jax.vmap(
          functools.partial(
              rlax.transformed_n_step_q_learning,
              n=self.bootstrap_n),
          in_axes=(2, None, 2, None, 1, None), out_axes=1)

        td_error = td_error_fn(
          online_sf[:-1],       # [T, A, C] (vmap 2) 
          online_actions[:-1],  # [T]       (vmap None) 
          target_sf[1:],        # [T, A, C] (vmap 2) 
          selector_actions[1:], # [T]       (vmap None) 
          cumulants[:cum_idx],       # [T, C]    (vmap 1) 
          discounts[:-1])       # [T]       (vmap None)

      elif self.loss == "transformed_q_lambda":
        td_error_fn = jax.vmap(
          functools.partial(
              rlax.transformed_q_lambda,
              lambda_=self.lambda_,
              tx_pair=self.tx_pair),
          in_axes=(2, None, 1, None, 2), out_axes=1)

        td_error = td_error_fn(
          online_sf[:-1],       # [T, A, C] (vmap 2)
          online_actions[:-1],  # [T]       (vmap None)
          cumulants[:cum_idx],       # [T, C]    (vmap 1)
          discounts[:-1],       # [T]       (vmap None)
          target_sf[1:],        # [T, A, C] (vmap 2)
        )
      elif self.loss == "q_lambda":
        td_error_fn = jax.vmap(
          functools.partial(
              rlax.q_lambda,
              lambda_=self.lambda_),
          in_axes=(2, None, 1, None, 2), out_axes=1)

        td_error = td_error_fn(
          online_sf[:-1],       # [T, A, C] (vmap 2)
          online_actions[:-1],  # [T]       (vmap None)
          cumulants[:cum_idx],       # [T, C]    (vmap 1)
          discounts[:-1],       # [T]       (vmap None)
          target_sf[1:],        # [T, A, C] (vmap 2)
        )

      return td_error # [T, C]


    # ======================================================
    # Prepare loss (via vmaps)
    # ======================================================
    # vmap over batch dimension (B)
    sf_loss = jax.vmap(sf_loss, in_axes=1, out_axes=1)
    # vmap over policy dimension (N)
    sf_loss = jax.vmap(sf_loss, in_axes=(2, None, 2, 2, None, None), out_axes=2)
    # output = [0=T, 1=B, 2=N, 3=C]
    batch_td_error = sf_loss(
      online_sf,        # [T, B, N, A, C] (vmap 2,1)
      online_actions,   # [T, B]          (vmap None,1)
      target_sf,        # [T, B, N, A, C] (vmap 2,1)
      selector_actions, # [T, B, N]       (vmap 2,1)
      cumulants,        # [T, B, C]       (vmap None,1)
      discounts)        # [T, B]          (vmap None,1)


    if self.mask_loss:
      # [T, B]
      episode_mask = make_episode_mask(data, include_final=False)
      # average over {T, N, C} --> # [B]
      batch_loss = episode_mean(
        x=(0.5 * jnp.square(batch_td_error)).mean(axis=(2,3)),
        mask=episode_mask[:-1])
    else:
      batch_loss = (0.5 * jnp.square(batch_td_error)).mean(axis=(0,2,3))

    batch_td_error = batch_td_error.mean(axis=(2, 3)) # [T, B]

    metrics = {
      f'z.loss_SfMain_{self.loss}': batch_loss.mean(),
      'z.sf_mean': online_sf.mean(),
      'z.sf_var': online_sf.var(),
      'z.sf_max': online_sf.max(),
      'z.sf_min': online_sf.min()}

    return batch_td_error, batch_loss, metrics # [T, B], [B]
