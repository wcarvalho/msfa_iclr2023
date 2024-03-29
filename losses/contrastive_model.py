import numpy as np
import jax.numpy as jnp
import distrax
import haiku as hk
import jax
from losses.utils import episode_mean, make_episode_mask

def normalize(x):
  return  x / (1e-5+jnp.linalg.norm(x, axis=-1, keepdims=True))


class OldModuleContrastLoss:
  """"""
  def __init__(self, coeff: float, temperature: float = 0.01, extra_negatives: int = 10, prediction='delta'):
    self.coeff = coeff
    self.temperature = temperature
    self.extra_negatives = extra_negatives
    self.prediction = prediction.lower()
    assert self.prediction in ['delta', 'state']


  def __call__(self, data, online_preds, online_state, **kwargs):

    model_preds = online_preds.model_outputs  # [T, B, N, D]

    state = online_preds.memory_out.hidden[:-1]  # [T, B, N, D]
    next_state = online_preds.memory_out.hidden[1:]  # [T, B, N, D]

    if self.prediction == 'delta':
      prediction = model_preds
      label = next_state - state
    elif self.prediction == 'state':
      prediction = state + model_preds
      label = next_state

    # -----------------------
    # L2 norm
    # -----------------------
    # [T-1, B, N, D]
    prediction = prediction / (1e-5+jnp.linalg.norm(prediction, axis=-1, keepdims=True))
    label = label / (1e-5+jnp.linalg.norm(label, axis=-1, keepdims=True))

    def contrastive_loss(anchors, positives):
      # logits

      logits = jnp.matmul(anchors, positives.transpose(0, 2, 1)) # [B, N, N]
      B, N = logits.shape[:2]

      identity = jnp.identity(logits.shape[1]).astype(jnp.float32)
      identity = jnp.expand_dims(identity, axis=0)
      positive_logits = identity*logits
      negative_logits = (1 - identity)*logits

      all_logits = logits
      more_negative_logits = negative_logits
      if self.extra_negatives > 0:
        # add extra negatives
        negative_options = positives.reshape(-1, positives.shape[-1])

        # get extra_negatives for each batch
        nnegatives = self.extra_negatives*B
        negative_idx = np.random.randint(nnegatives, size=nnegatives)
        # [B, N, E, D]
        negatives = negative_options[negative_idx].reshape(B, self.extra_negatives, -1)
        more_negative_logits = jnp.matmul(anchors, negatives.transpose(0, 2, 1))

        all_logits = jnp.concatenate((logits, more_negative_logits), axis=-1)
        all_logits = all_logits + 1e-5

      all_logits = all_logits/jnp.exp(self.temperature)

      labels = jnp.arange(N)

      likelihood = distrax.Categorical(logits=all_logits).log_prob(labels)

      return positive_logits, negative_logits, more_negative_logits, likelihood


    pos_logits, mod_neg_logits, rand_neg_logits, likelihood = hk.BatchApply(contrastive_loss)(prediction, label)


    # output is [B]
    batch_loss = episode_mean(
      x=(-likelihood).mean(-1),
      mask=make_episode_mask(data)[:-1]).mean()

    T, B, N = label.shape[:3]
    metrics = {
      'loss_contrast': batch_loss,
      'z.contrast.neg_mod_logits' : mod_neg_logits.mean(),
      'z.contrast.neg_ran_logits' : rand_neg_logits.mean(),
      'z.contrast.pos_logits' : pos_logits.sum()/(T*B*N),
    }
    return self.coeff*batch_loss, metrics

class ModuleContrastLoss:
  """"""
  def __init__(self, coeff: float, temperature: float = 0.01, extra_negatives: int = 10, prediction='delta'):
    self.coeff = coeff
    self.temperature = temperature
    self.extra_negatives = extra_negatives
    self.prediction = prediction.lower()
    assert self.prediction in ['delta', 'state']


  def __call__(self, data, online_preds, online_state, **kwargs):

    model_preds = online_preds.model_outputs  # [T, B, N, D]
    state = online_preds.memory_out.hidden[:-1]  # [T, B, N, D]
    next_state = online_preds.memory_out.hidden[1:]  # [T, B, N, D]

    if self.prediction == 'delta':
      prediction = model_preds
      label = next_state - state
    elif self.prediction == 'state':
      prediction = state + model_preds
      label = next_state

    # -----------------------
    # L2 norm
    # -----------------------
    # [T-1, B, N, D]
    prediction = prediction / (1e-5+jnp.linalg.norm(prediction, axis=-1, keepdims=True))
    label = label / (1e-5+jnp.linalg.norm(label, axis=-1, keepdims=True))

    def contrastive_loss(predictions : jnp.ndarray, labels : jnp.ndarray, negatives : jnp.ndarray):
      """Summary
      
      Args:
          predictions (jnp.ndarray): N x D
          labels (jnp.ndarray): N x D
          negatives (jnp.ndarray): T x B x N x D
      
      Returns:
          TYPE: Description
      """
      logits = jnp.matmul(predictions, labels.transpose(1, 0)) # [N, N]

      identity = jnp.identity(logits.shape[1]).astype(jnp.float32)
      identity = jnp.expand_dims(identity, axis=0)
      positive_logits = identity*logits
      negative_logits = (1 - identity)*logits

      N = logits.shape[0]
      all_logits = logits
      more_negative_logits = negative_logits

      if self.extra_negatives > 0:
        # add extra negatives
        D = negatives.shape[-1]
        negatives = negatives.reshape(-1, D)

        # get extra_negatives for each batch
        nnegatives = self.extra_negatives*N # M*N
        negative_idx = np.random.randint(len(negatives), size=nnegatives)

        _negatives = negatives[negative_idx].reshape(N, self.extra_negatives, -1)

        dot = lambda a,b : jnp.matmul(a, b.transpose(1, 0))
        more_negative_logits = jax.vmap(dot)(predictions, _negatives) # N x M

        all_logits = jnp.concatenate((logits, more_negative_logits), axis=-1)
        all_logits = all_logits + 1e-5

      all_logits = all_logits/jnp.exp(self.temperature)


      labels = jnp.arange(N)

      likelihood = distrax.Categorical(logits=all_logits).log_prob(labels)

      return positive_logits, negative_logits, more_negative_logits, likelihood


    contrastive_loss = jax.vmap(contrastive_loss, in_axes=(0, 0, None), out_axes=0) # N
    contrastive_loss = jax.vmap(contrastive_loss, in_axes=(1, 1, None), out_axes=1) # B


    pos_logits, mod_neg_logits, rand_neg_logits, likelihood = contrastive_loss(prediction, label, label)


    # output is [B]
    batch_loss = episode_mean(
      x=(-likelihood).mean(-1),
      mask=make_episode_mask(data)[:-1]).mean()

    T, B, N = label.shape[:3]
    metrics = {
      'loss_contrast': batch_loss,
      'z.contrast.neg_mod_logits' : mod_neg_logits.mean(),
      'z.contrast.neg_ran_logits' : rand_neg_logits.mean(),
      'z.contrast.pos_logits' : pos_logits.sum()/(T*B*N),
    }
    return self.coeff*batch_loss, metrics

class TimeContrastLoss:
  """"""
  def __init__(self, coeff: float,
    temperature: float = 0.01,
    extra_negatives: int = 0,
    normalize_step: bool=False):
    self.coeff = coeff
    self.temperature = temperature
    self.extra_negatives = extra_negatives
    self.normalize_step = normalize_step

  def __call__(self, data, online_preds, online_state, **kwargs):

    model_preds = online_preds.model_outputs  # [T, B, N, D]

    state = online_preds.memory_out.hidden[:-1]  # [T, B, N, D]
    next_state = online_preds.memory_out.hidden[1:]  # [T, B, N, D]

    # -----------------------
    # L2 norm
    # -----------------------
    # [T-1, B, N, D]
    labels = normalize(next_state)   # positive
    if self.normalize_step:
      state = normalize(state)
      model_preds = normalize(model_preds)
    predictions = normalize(state + model_preds)  # anchor


    def contrastive_loss(
      predictions: jnp.ndarray,
      labels: jnp.ndarray):
      """Summary
      
      Args:
          predictions (jnp.ndarray): N x D
          labels (jnp.ndarray): N x D
          incorrect (jnp.ndarray): N x D
      """


      logits = jnp.matmul(predictions, labels.transpose(1, 0)) # [N, N]

      positive_logits = jnp.diagonal(logits, axis1=0, axis2=1) # N, N
      identity = jnp.identity(logits.shape[0]).astype(jnp.float32)
      negative_logits = (1 - identity)*logits

      all_logits = logits
      if self.extra_negatives:
        negative_idx = np.random.randint(self.extra_negatives, size=self.extra_negatives)
        negatives = labels[negative_idx] # N
        more_negative_logits = jnp.matmul(predictions, negatives.transpose(1, 0))

        all_logits = jnp.concatenate((logits, more_negative_logits), axis=-1)

        negative_logits = jnp.concatenate((negative_logits, more_negative_logits), axis=-1)

      all_logits = all_logits/jnp.exp(self.temperature)
      all_logits = all_logits + 1e-5

      label_idxs = jnp.arange(logits.shape[0])

      log_prob = distrax.Categorical(logits=all_logits).log_prob(label_idxs)

      return positive_logits, negative_logits, log_prob

    contrastive_loss = jax.vmap(contrastive_loss, in_axes=1, out_axes=1) # B
    contrastive_loss = jax.vmap(contrastive_loss, in_axes=2, out_axes=2) # N
    positive_logits, negative_logits, log_prob = contrastive_loss(predictions,
      labels)

    # input is [T, B, N, D]
    # output is [B, N]
    batch_loss = episode_mean(
      x=(-log_prob).mean(-1),
      mask=make_episode_mask(data)[:-1]).mean()

    metrics = {
      'loss_time_contrast': batch_loss,
      # 'z.time_contrast_prob': jnp.log(log_prob).mean(),
      'z.time.positive_logits' : positive_logits.mean(),
      'z.time.negative_logits' : negative_logits.mean(),
      'z.time.model_preds_mean' : model_preds.mean(),
      'z.time.model_preds_var' : model_preds.var(),
      'z.time.state_mean' : state.mean(),
      'z.time.state_var' : state.var(),
    }

    return self.coeff*batch_loss, metrics
