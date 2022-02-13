import jax.numpy as jnp
import distrax

class VaeAuxLoss:
  """Copied from distrax example:
    https://github.com/deepmind/distrax/blob/master/examples/vae.py

    Changes: added beta for beta-vae.
    """
  def __init__(self, coeff: float, beta: float=25):
    self.coeff = coeff
    self.beta = beta

  def __call__(self, data, online_preds, **kwargs):

    mean = online_preds.mean
    std = online_preds.std
    logits = online_preds.reconstruction
    image = data.observation.observation.image.astype(mean.dtype)/255.0

    # -----------------------
    # likelihood
    # -----------------------
    likelihood_distrib = distrax.Independent(
        distrax.Bernoulli(logits=logits),
        reinterpreted_batch_ndims=len(image.shape[2:]))
    log_likelihood = likelihood_distrib.log_prob(image)
    log_likelihood = log_likelihood/np.prod(image.shape[2:])

    # -----------------------
    # kl
    # -----------------------
    prior_z = distrax.MultivariateNormalDiag(
        loc=jnp.zeros(mean.shape),
        scale_diag=jnp.ones(mean.shape))
    variational_distrib = distrax.MultivariateNormalDiag(
        loc=mean, scale_diag=std)

    kl = variational_distrib.kl_divergence(prior_z)

    # -----------------------
    # elbo
    # -----------------------
    elbo = log_likelihood - self.beta*kl

    batch_loss = -jnp.mean(elbo)

    metrics = {
      'loss_vae': batch_loss,
      'z.vae.kl' : kl.mean(),
      'z.vae.recon': -log_likelihood.mean()}
    return self.coeff*batch_loss, metrics
