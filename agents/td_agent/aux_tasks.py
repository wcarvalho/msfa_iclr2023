import jax.numpy as jnp

def cumulant_from_reward(
  data, online_preds, online_state, target_preds, target_state, 
  coeff=.01):
  """Learn cumulants via ||r - dot(w, cumulants) ||
  """
  rewards = data.reward  # ground-truth

  cumulants = online_preds.cumulants  # predicted
  task = data.observation.observation.task  # ground-truth
  reward_pred = jnp.sum(cumulants*task, -1)


  error = jnp.power(reward_pred - rewards, 2).sum(-1).mean()

  return error*coeff, dict(reward_loss=error)

