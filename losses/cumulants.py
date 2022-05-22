import distrax
import jax
import jax.numpy as jnp
import haiku as hk
import rlax

class CumulantRewardLoss:
  """"""
  def __init__(self, coeff: float, loss: str = 'l2', shorten_data_for_cumulant: bool = False,
    balance: float = 0,
    reward_bias: float = 0,
    nmodules: int = 1,
    l1_coeff=None,
    wl1_coeff=None):
    self.coeff = coeff
    loss = loss.lower()
    assert loss in ['l2', 'binary']
    self.loss = loss
    self.shorten_data_for_cumulant = shorten_data_for_cumulant
    self.balance = balance
    self.random = True
    self.l1_coeff = l1_coeff
    self.reward_bias = reward_bias
    self.nmodules = nmodules
    self.wl1_coeff = wl1_coeff


  def __call__(self, data, online_preds, key, **kwargs):
    cumulants = online_preds.cumulants  # predicted  [T, B, D]
    task = online_preds.w  # ground-truth  [T, B, D]
    not_done = data.discount[:-1]  # predicted  [T, B, D]
    cumulants = cumulants*jnp.expand_dims(not_done, axis=2)

    rewards = data.reward  # ground-truth  [T, B]
    if self.reward_bias:
      rewards = rewards + self.reward_bias # offset time-step penality

    if self.shorten_data_for_cumulant and cumulants.shape[0] < task.shape[0]:
      shape = cumulants.shape[0]
      task = task[:shape]
      rewards = rewards[:shape]


    reward_pred = jnp.sum(cumulants*task, -1)  # dot product  [T, B]
    if self.loss == 'l2':
      error = rlax.l2_loss(predictions=reward_pred, targets=rewards)
    elif self.loss == 'binary':
      error = -distrax.Bernoulli(logits=reward_pred).log_prob(rewards)

    not_done = not_done.reshape(-1)
    error = error.reshape(-1)
    error = error*not_done

    if self.balance > 0:
      # flatten
      raw_final_error = error.mean()
      nonzero = (rewards != 0).reshape(-1)
      nonzero = nonzero.astype(jnp.float32)

      # get all that have 0 reward and 
      zero = (jnp.abs(rewards) < 1e-5).reshape(-1)

      valid = not_done > 0
      zero = jnp.logical_and(zero, valid)
      probs = zero.astype(jnp.float32)*self.balance
      keep_zero = distrax.Independent(
        distrax.Bernoulli(probs=probs)).sample(seed=key)
      keep_zero = keep_zero.astype(jnp.float32)


      all_keep = (nonzero + keep_zero)
      total_keep = all_keep.sum()
      final_error = (error*all_keep).sum()/(1e-5+total_keep)
      positive_error = (error*nonzero).sum()/(1e-5+nonzero.sum())

      metrics = {
        f'loss_reward_{self.loss}': final_error,
        f'z.raw_loss_{self.loss}': raw_final_error,
        f'z.positive_error': positive_error,
        f'z.reward_pred': reward_pred.mean(),
        f'z.task': task.mean(),
        f'z.phi': cumulants.mean(),
        f'z.task_std': task.std(),
        f'z.phi_std': cumulants.std(),
      }
    else:
      final_error = error.mean()
      metrics = {
        f'loss_reward_{self.loss}': final_error,
      }

    final_error = final_error*self.coeff
    if self.l1_coeff is not None and self.l1_coeff != 0:

      if self.nmodules > 1:
        cumulants = jnp.stack(jnp.split(cumulants, self.nmodules, axis=-1), axis=2)
      phi_l1 = jnp.linalg.norm(cumulants, ord=1, axis=-1)
      phi_l1 = phi_l1.mean()

      metrics['loss_phi_l1'] = phi_l1
      phi_l1 = phi_l1*self.l1_coeff

      final_error = final_error + phi_l1

    if self.wl1_coeff is not None and self.wl1_coeff != 0:

      w_l1 = jnp.linalg.norm(task, ord=1, axis=-1)
      w_l1 = w_l1.mean()

      metrics['loss_w_l1'] = w_l1
      w_l1 = w_l1*self.wl1_coeff

      final_error = final_error + w_l1

    return final_error, metrics



class CumulantCovLoss:
  """"""
  def __init__(self, coeff: float, blocks: int=None, loss: bool=False):
    self.coeff = coeff
    self.blocks = blocks
    self.loss = loss.lower()
    assert self.loss in ['mean', 'l1', 'l2']

  def __call__(self, data, online_preds, **kwargs):
    cumulants = online_preds.cumulants  # predicted  [T, B, D]
    not_done = data.discount[:-1]  # predicted  [T, B, D]
    cumulants = cumulants*jnp.expand_dims(not_done, axis=2)

    # -----------------------
    # setup
    # -----------------------
    dim = cumulants.shape[-1]
    block_size = dim//self.blocks
    assert dim % self.blocks == 0
    ones = jnp.ones((block_size, block_size)).astype(cumulants.dtype)
    block_id = jax.scipy.linalg.block_diag(*[ones for _ in range(self.blocks)])

    # -----------------------
    # compute
    # -----------------------
    def cov_diag(x: jnp.ndarray,
      block_id: jnp.ndarray):
      """Summary
      
      Args:
          x (jnp.ndarray): T x D
          weights (jnp.ndarray): T x D
          block_id (jnp.ndarray): D x D
      
      Returns:
          TYPE: Description
      """
      cov = jnp.cov(x, rowvar=False) # [D, D]
      cov_on = cov*block_id
      cov_off = cov - cov_on
      return cov, cov_off, cov_on

    cov, cov_off, cov_on = jax.vmap(cov_diag, in_axes=(1, None), out_axes=0)(cumulants, block_id)

    cov_off_mean = cov_off.mean()
    cov_on_mean = cov_on.mean()

    if self.loss == "mean":
      loss = cov_off_mean
    elif self.loss == "l1":
      loss = jnp.linalg.norm(cov_off, ord=1, axis=(1,2)).mean()
    elif self.loss == "l2":
      loss = jnp.linalg.norm(cov_off, ord=2, axis=(1,2)).mean()

    metrics = {
      f'loss_cov_{self.loss}': loss,
      "cov" : cov.mean(),
      "cov_off" : cov_off_mean,
      "cov_on" :cov_on_mean,}

    return loss*self.coeff, metrics
