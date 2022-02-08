from typing import NamedTuple, Optional, Tuple

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp

def flatten_conv(x):
  return jnp.reshape(x, [*x.shape[:-3], -1]) # flatten last 3: H, W, C



class VaeEncoderOutputs(NamedTuple):
  """
  samples: latent samples
  mean: mean of samples (gaussian)
  std: std of samples (gaussian)
  conv_out: output of encoder conv network (pre flattening)
  reconstruction:  reconstruction using latent
  """
  samples: jnp.ndarray
  mean: jnp.ndarray
  std: jnp.ndarray
  conv_out: jnp.ndarray


class VAE(hk.Module):
  def __init__(self, latent_dim=128, latent_source='memory',
    encoder=None, decoder=None
    ):
    super().__init__(name='encoder')

    # -----------------------
    # networks
    # -----------------------
    if encoder is None:
      encoder = hk.Sequential([
          hk.Conv2D(32, [8, 8], 4),
          jax.nn.relu,
          hk.Conv2D(64, [4, 4], 2),
          jax.nn.relu,
          hk.Conv2D(64, [3, 3], 2),
          jax.nn.relu,
          hk.Conv2D(16, [1, 1], 1),
      ])
    self.encode = encoder

    self.z_mean = hk.Linear(latent_dim)
    self.z_logstd = hk.Linear(latent_dim)

    if decoder is None:
      decoder = hk.Sequential([
          hk.Conv2DTranspose(64, [1, 1], 1, padding='SAME'),
          jax.nn.relu,
          hk.Conv2DTranspose(64, [4, 4], 2, padding='VALID'),
          jax.nn.relu,
          hk.Conv2DTranspose(32, [4, 4], 2, padding='VALID'),
          jax.nn.relu,
          hk.Conv2DTranspose(3, [6, 6], 2, padding='VALID'),
      ])
    self.decoder = decoder

    # -----------------------
    # settings
    # -----------------------
    latent_source = latent_source.lower()
    assert latent_source in ['memory', 'samples']
    self.latent_source = latent_source

  def bottleneck(self, mean, key) -> jnp.ndarray:

    return samples, mean, std

  def decode(self, latent, input_dims, conv_dims) -> jnp.ndarray:
    latent = hk.Linear(np.prod(conv_dims))(latent)
    latent = latent.reshape(-1, *conv_dims)

    pred = self.decoder(latent)

    Hx, Wx = input_dims[:2]
    assert Hx <= pred.shape[1] and Wx <= pred.shape[2], f"{(Hx, Wx)} vs {(pred.shape[1], pred.shape[2])}"

    pred = pred[:, :Hx, :Wx]
    return pred

  def __call__(self, inputs):
    """Used at inference time to get latent representation.
    """
    conv_out = self.encode(inputs)
    conv_flat = flatten_conv(conv_out)

    # -----------------------
    # mean + std
    # -----------------------
    mean = self.z_mean(conv_flat)
    logstd = self.z_logstd(conv_flat)
    std = jnp.exp(logstd)

    # -----------------------
    # sample
    # -----------------------
    samples = mean + std * jax.random.normal(hk.next_rng_key(), mean.shape)

    return VaeEncoderOutputs(
      conv_out=conv_out,
      samples=samples,
      mean=mean,
      std=std)

  def aux_task(self, inputs, obs, memory_out, **kwargs):
    input_dims = inputs.observation.image.shape[1:]
    conv_dims = obs.conv_out.shape[1:]

    if self.latent_source == "memory":
      latent = memory_out
    elif self.latent_source == "samples":
      latent = obs.samples

    reconstruction = self.decode(memory_out, input_dims, conv_dims)
    return dict(
      samples=obs.samples,
      mean=obs.mean,
      std=obs.std,
      reconstruction=reconstruction,
      )
