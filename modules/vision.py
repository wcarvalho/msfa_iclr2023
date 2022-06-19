"""Vision Modules."""

from acme.jax.networks import base
import haiku as hk
import jax
import jax.numpy as jnp
from utils import vmap

Images = jnp.ndarray


class AtariVisionTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, flatten=True, conv_dim = 16, out_dim=0):
    super().__init__(name='atari_torso')
    layers = [
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
    ]
    if conv_dim:
      layers.append(hk.Conv2D(conv_dim, [1, 1], 1))
    self._network = hk.Sequential(layers)

    self.flatten = flatten
    if out_dim:
      self.out_net = hk.Linear(out_dim)
    else:
      self.out_net = lambda x:x

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return self.out_net(flat)


class BabyAIVisionTorso(base.Module):
  """Convolutional stack used in BabyAI codebase."""

  def __init__(self, flatten=True, conv_dim=16, out_dim=0):
    super().__init__(name='babyai_torso')
    layers = [
        hk.Conv2D(128, [8, 8], stride=8),
        hk.Conv2D(128, [3, 3], stride=1),
        jax.nn.relu,
        hk.Conv2D(128, [3, 3], stride=1),
        jax.nn.relu,
        hk.Conv2D(conv_dim, [1, 1], stride=1),
    ]
    self._network = hk.Sequential(layers)

    self.flatten = flatten
    if out_dim:
      self.out_net = hk.Linear(out_dim)
    else:
      self.out_net = lambda x:x

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return self.out_net(flat)


class BabyAIymbolicVisionTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, num_symbols, flatten=True, conv_dim = 16, out_dim=0):
    super().__init__(name='atari_torso')

    self.embedder_fn = lambda: hk.Embed(vocab_size=num_symbols, embed_dim=conv_dim)
    self.flatten = flatten
    self.conv_dim = conv_dim

  def __call__(self, inputs) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs.image)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


    embedded = vmap.batch_multihead(
      fn=self.embedder_fn,
      x=inputs.image.astype(jnp.int32),
      wrap_vmap=lambda fn: hk.BatchApply(fn, num_dims=3),
      vmap='lift',
    )

    # flatten channels together
    embedded = embedded.reshape(*embedded.shape[:inputs_rank-1], -1)

    # linear over them
    embedded = hk.Conv2D(self.conv_dim, [1, 1], stride=1)(embedded)
    embedded = jax.nn.relu(embedded)

    if not self.flatten:
      return embedded

    if batched_inputs:
      flat = jnp.reshape(embedded, [embedded.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(embedded, [-1])  # [D]

    return flat