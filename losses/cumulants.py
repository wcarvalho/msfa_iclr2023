import distrax
import jax
import jax.numpy as jnp
import rlax

class CumulantRewardLoss:
  """"""
  def __init__(self, coeff: float, loss: str = 'l2', shorten_data_for_cumulant: bool = False, balance: bool = False):
    self.coeff = coeff
    loss = loss.lower()
    assert loss in ['l2', 'binary']
    self.loss = loss
    self.shorten_data_for_cumulant = shorten_data_for_cumulant
    self.balance = balance

  def __call__(self, data, online_preds, **kwargs):
    cumulants = online_preds.cumulants  # predicted  [T, B, D]
    task = online_preds.w  # ground-truth  [T, B, D]

    rewards = data.reward  # ground-truth  [T, B]

    if self.shorten_data_for_cumulant and cumulants.shape[0] < task.shape[0]:
      shape = cumulants.shape[0]
      task = task[:shape]
      rewards = rewards[:shape]


    reward_pred = jnp.sum(cumulants*task, -1)  # dot product  [T, B]
    if self.loss == 'l2':
      error = rlax.l2_loss(predictions=reward_pred, targets=rewards)
    elif self.loss == 'binary':
      error = -distrax.Bernoulli(logits=reward_pred).log_prob(rewards)

    if self.balance:
      raise RuntimeError
      # error = error.reshape(-1)
      # import ipdb; ipdb.set_trace()
      # nonzero = rewards != 0 
      # num_nonzero = nonzero.sum()

    else:
      error = error.mean()

    metrics = {
      f'loss_reward_{self.loss}': error,
    }
    return error*self.coeff, metrics
