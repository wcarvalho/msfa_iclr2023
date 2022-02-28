from typing import Sequence, Optional

import haiku as hk
import jax.numpy as jnp

class DuellingSfQNet(hk.Module):
  """A Duelling MLP SF-network."""

  def __init__(
      self,
      num_actions: int,
      num_cumulants: int,
      hidden_sizes: Sequence[int],
      w_init: Optional[hk.initializers.Initializer] = None,
  ):
    super().__init__(name='duelling_qsf_network')
    self.num_actions = num_actions
    self.num_cumulants = num_cumulants

    self._value_mlp = hk.nets.MLP([*hidden_sizes, num_cumulants], w_init=w_init)
    self._advantage_mlp = hk.nets.MLP([*hidden_sizes, num_actions*num_cumulants],
                                      w_init=w_init)

  def __call__(self, inputs: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Forward pass of the duelling network.
    
    Args:
        inputs (jnp.ndarray): B x Z
        w (jnp.ndarray): B x A x C

    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Compute value & advantage for duelling.
    value = self._value_mlp(inputs)  # [B, C]
    advantages = self._advantage_mlp(inputs)  # [B, A*C]

    value = jnp.expand_dims(value, axis=1) # [B, 1, C]
    advantages = jnp.reshape(advantages,
      [advantages.shape[0], self.num_actions, self.num_cumulants]) # [B, A, C]

    # Advantages have zero mean.
    advantages -= jnp.mean(advantages, axis=1, keepdims=True)  # [B, A, C]


    sf = value + advantages  # [B, A, C]

    q_values = jnp.sum(sf*w, axis=-1) # [B, A]

    return sf, q_values