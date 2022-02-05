"""Vision Modules."""

from acme.jax.networks import base
import haiku as hk
import jax
import jax.numpy as jnp


Images = jnp.ndarray


class AtariVisionTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, flatten=True):
    super().__init__(name='atari_torso')
    self._network = hk.Sequential([
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
        hk.Conv2D(16, [1, 1], 1)
    ])

    self.flatten = flatten

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    return jnp.reshape(outputs, [-1])  # [D]
