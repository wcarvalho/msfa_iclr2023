import jax.numpy as jnp

def episode_mean(x, done):
  if len(done.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(done.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    batch_loss = jnp.multiply(x, jnp.expand_dims(done, dims))
  else:
    batch_loss = jnp.multiply(x, done)
  return (batch_loss.sum(0))/(done.sum(0)+1e-5)
