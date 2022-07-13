from typing import Callable, Optional, Sequence, NamedTuple


import haiku as hk
import jax.numpy as jnp
import jax

class DotQMlpOutputs(NamedTuple):
  q: jnp.ndarray  # Q-values

class DotQMlpInputs(NamedTuple):
  inputs: jnp.ndarray  # task
  task: jnp.ndarray  # task

class DotQMlp(hk.Module):
  """A Duelling MLP SF-network."""

  def __init__(
      self,
      num_actions: int,
      task_dim: int,
      hidden_sizes: Sequence[int],
      w_init: Optional[hk.initializers.Initializer] = None,
  ):
    super().__init__(name='dot_qhead')
    self.num_actions = num_actions
    self.task_dim = task_dim

    self.mlp = hk.nets.MLP([*hidden_sizes, num_actions*task_dim], w_init=w_init)


  def __call__(self, inputs: DotQMlpInputs, **kwargs) -> jnp.ndarray:
    """Forward pass of the duelling network.
    
    Args:
        inputs (jnp.ndarray): B x Z
        w (jnp.ndarray): B x W

    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Compute value & advantage for duelling.
    outputs = self.mlp(inputs.inputs)  # [B, A*C]

    # [B, A, C]
    outputs = jnp.reshape(outputs,
      [outputs.shape[0], self.num_actions, self.task_dim])

    q_vals = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)(outputs, inputs.task)

    # [B, A]
    q_vals = q_vals.sum(-1)

    return DotQMlpOutputs(q=q_vals)