import jax.numpy as jnp
import distrax
import rlax

def cumulant_from_reward(
  data, online_preds, online_state, target_preds, target_state, 
  coeff=.01, loss='binary'):
  """Learn cumulants
  
  Args:
      coeff (float, optional): coefficient for loss
      method (str, optional): l2 = minimize L2(r, dot(w, cumulants)). 
        Binary = binary cross entropy
  """
  loss = loss.lower()
  assert loss in ['l2', 'binary']

  cumulants = online_preds.cumulants  # predicted  [T, B, D]
  task = data.observation.observation.task  # ground-truth  [T, B, D]
  reward_pred = jnp.sum(cumulants*task, -1)  # dot product  [T, B]


  rewards = data.reward  # ground-truth  [T, B]

  if loss == 'l2':
    error = rlax.l2_loss(predictions=reward_pred, targets=rewards).mean()

  if loss == 'binary':
    error = -distrax.Bernoulli(logits=reward_pred).log_prob(rewards).mean()

  metrics = {
    f'loss_reward_{loss}': error,
    'batch_reward' : rewards.mean(),
  }
  return error*coeff, metrics


def vae_loss(
  data, online_preds, online_state, target_preds, target_state, 
  coeff=.01, loss='binary'):
  """Learn cumulants
  
  Args:
      coeff (float, optional): coefficient for loss
      method (str, optional): l2 = minimize L2(r, dot(w, cumulants)). 
        Binary = binary cross entropy
  """
  loss = loss.lower()
  assert loss in ['l2', 'binary']

  cumulants = online_preds.cumulants  # predicted  [T, B, D]
  task = data.observation.observation.task  # ground-truth  [T, B, D]
  reward_pred = jnp.sum(cumulants*task, -1)  # dot product  [T, B]


  rewards = data.reward  # ground-truth  [T, B]

  if loss == 'l2':
    error = rlax.l2_loss(predictions=reward_pred, targets=rewards).mean()

  if loss == 'binary':
    error = -distrax.Bernoulli(logits=reward_pred).log_prob(rewards).mean()

  metrics = {
    f'loss_reward_{loss}': error,
    'batch_reward' : rewards.mean(),
  }
  return error*coeff, metrics

