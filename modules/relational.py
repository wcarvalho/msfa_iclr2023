"""Vision Modules."""

from acme.jax.networks import base
import haiku as hk
import jax
import jax.numpy as jnp
from acme.jax import utils


class RelationalLayer(base.Module):
  """Multi-head attention based relational layer."""

  def __init__(self,
    num_heads=2,
    w_init_scale=2.,
    position_embed=32,
    key_size=64,
    residual=True,
    shared_parameters=True, name='relational'):
    super().__init__(name=name)
    self.shared_parameters = shared_parameters
    self.num_heads = num_heads
    self.w_init_scale = w_init_scale
    self.position_embed = position_embed
    self.residual = residual
    self.key_size = key_size

  def __call__(self, factors: jnp.ndarray, queries: jnp.ndarray=None) -> jnp.ndarray:
    """Summary
    
    Args:
        factors (jnp.ndarray): B x N x D
        queries (jnp.ndarray, optional): B x N x D. If not given, factors are treated as queries.
    
    Returns:
        jnp.ndarray: Description
    """
    if queries is None: 
      queries = factors

    if self.shared_parameters:
      values = self.shared_attn(queries, factors)
    else:
      values = self.independent_attn(queries, factors)

    if self.residual:
      return factors + values
    else:
      raise NotImplementedError

  def prepare_data(
      self,
      queries: jnp.ndarray,
      factors: jnp.ndarray,
      ) -> jnp.ndarray:
    """ Multihead attention expects [N, B, D]. Fix. """
    # -----------------------
    # prepare data
    # -----------------------
    # convert things to dims expected by multihead attention
    factors = factors.transpose((1,0,2))  # [N, B, D]
    queries = queries.transpose((1,0,2))  # [N, B, D]

    N, B = queries.shape[:2]
    D = factors.shape[2]

    # add zeros for no attention
    zeros = jnp.zeros((1, B, D))
    factors = jnp.concatenate((factors, zeros)) # [N+1, B, D]

    return queries, factors

  def shared_attn(
      self,
      queries: jnp.ndarray,
      factors: jnp.ndarray,
      ) -> jnp.ndarray:
    """Share parameters across queries.
    
    Args:
        queries (jnp.ndarray): [B, N, D]
        factors (jnp.ndarray): [B, N, D]
    
    Returns:
        jnp.ndarray: Description
    """
    queries, factors = self.prepare_data(queries, factors)
    N, B, D = factors.shape

    if self.position_embed:
      self.embedder = hk.Embed(
        vocab_size=N,
        embed_dim=self.position_embed)
      embeddings = self.embedder(jnp.arange(N)) # N x D
      concat = lambda a,b: jnp.concatenate((a, b), axis=-1)
      factors = jax.vmap(concat, in_axes=(1, None), out_axes=1)(
        factors, embeddings)

    attn = hk.MultiHeadAttention(
      num_heads=self.num_heads,
      key_size=self.key_size or D,
      model_size=D,
      w_init_scale=self.w_init_scale,
      )
    attn = jax.vmap(attn, in_axes=(0, None, None))

    out = attn(queries, factors, factors)
    out = out.transpose((1,0,2))  # [B, N, D]
    return out

  def independent_attn(
      self,
      queries: jnp.ndarray,
      factors: jnp.ndarray,
      ) -> jnp.ndarray:
    raise NotImplementedError()