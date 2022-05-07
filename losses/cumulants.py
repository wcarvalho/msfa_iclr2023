import distrax
import jax
import jax.numpy as jnp
import haiku as hk
import rlax

class CumulantRewardLoss:
  """"""
  def __init__(self, coeff: float, loss: str = 'l2', shorten_data_for_cumulant: bool = False,
    balance: float = 0,
    l1_coeff=None):
    self.coeff = coeff
    loss = loss.lower()
    assert loss in ['l2', 'binary']
    self.loss = loss
    self.shorten_data_for_cumulant = shorten_data_for_cumulant
    self.balance = balance
    self.random = True
    self.l1_coeff = l1_coeff

  def __call__(self, data, online_preds, key, **kwargs):
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


    if self.balance > 0:
      # flatten
      raw_final_error = error.mean()
      error = error.reshape(-1)
      nonzero = (rewards != 0).reshape(-1)
      nonzero = nonzero.astype(jnp.float32)

      # get all that have 0 reward and 
      zero = (jnp.abs(rewards) < 1e-5).reshape(-1)
      probs = zero.astype(jnp.float32)*self.balance
      keep_zero = distrax.Independent(
        distrax.Bernoulli(probs=probs)).sample(seed=key)
      keep_zero = keep_zero.astype(jnp.float32)


      all_keep = (nonzero + keep_zero)
      total_keep = all_keep.sum()
      final_error = (error*all_keep).sum()/(1e-5*total_keep)

      metrics = {
        f'loss_reward_{self.loss}': final_error,
        f'z.raw_loss_{self.loss}': raw_final_error,
        f'z.reward_pred': reward_pred.mean(),
        f'z.reward_pred_var': reward_pred.var(),
      }
    else:
      final_error = error.mean()
      metrics = {
        f'loss_reward_{self.loss}': final_error,
      }

    if self.l1_coeff is not None:
      phi_l1 = jnp.linalg.norm(cumulants, ord=1, axis=-1)
      phi_l1 = phi_l1.mean()
      metrics['loss_phi_l1'] = phi_l1

      final_error += phi_l1

    return final_error*self.coeff, metrics
