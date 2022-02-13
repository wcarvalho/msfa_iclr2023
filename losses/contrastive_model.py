import numpy as np
import jax.numpy as jnp
import distrax
import haiku as hk

class DeltaContrastLoss:
  """"""
  def __init__(self, coeff: float, temperature: float = 0.01, extra_negatives: int = 10):
    self.coeff = coeff
    self.temperature = temperature
    self.extra_negatives = extra_negatives

  def __call__(self, data, online_preds, online_state, **kwargs):

    delta_preds = online_preds.model_outputs  # [T, B, N, D]

    state = online_preds.memory_out[:-1]  # [T, B, N, D]
    next_state = online_preds.memory_out[1:]  # [T, B, N, D]
    delta = next_state - state

    # -----------------------
    # L2 norm
    # -----------------------

    delta = delta / jnp.linalg.norm(delta, axis=-1, keepdims=True)
    delta_preds = delta_preds / jnp.linalg.norm(delta_preds, axis=-1, keepdims=True)

    def contrastive_loss(anchors, positives):
      # logits

      logits = jnp.matmul(anchors, positives.transpose(0, 2, 1)) # [B, N, N]
      B, N = logits.shape[:2]

      # add extra negatives
      negative_options = positives.reshape(-1, positives.shape[-1])

      # get extra_negatives for each batch
      nnegatives = self.extra_negatives*B
      negative_idx = np.random.randint(nnegatives, size=nnegatives)
      # [B, N, E, D]
      negatives = negative_options[negative_idx].reshape(B, self.extra_negatives, -1)
      negative_logits = jnp.matmul(anchors, negatives.transpose(0, 2, 1))

      all_logits = jnp.concatenate((logits, negative_logits), axis=-1)
      all_logits = all_logits/jnp.exp(self.temperature)

      labels = jnp.arange(N)

      likelihood = distrax.Categorical(logits=all_logits).log_prob(labels)

      return logits, negative_logits, likelihood


    logits, negative_logits, likelihood = hk.BatchApply(contrastive_loss)(delta_preds, delta)

    positive_logits = jnp.diagonal(logits, axis1=2, axis2=3)

    batch_loss = -likelihood.mean()

    metrics = {
      'loss_contrast': batch_loss.mean(),
      'z.contrast.negative_logits' : negative_logits.mean(),
      'z.contrast.positive_logits' : positive_logits.mean(),
    }
    return self.coeff*batch_loss, metrics
