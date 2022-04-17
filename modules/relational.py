"""Vision Modules."""

from typing import Optional

from acme.jax.networks import base
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme.jax import utils

class MultiHeadAttention(hk.Module):
  """Multi-headed attention mechanism.
  As described in the vanilla Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      # TODO(romanring, tycai): migrate to a more generic `w_init` initializer.
      w_init_scale: float,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size
    self.w_init = hk.initializers.VarianceScaling(w_init_scale)

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Compute (optionally masked) MHA with queries, keys & values."""
    query_heads = self._linear_projection(query, self.key_size, "query")
    key_heads = self._linear_projection(key, self.key_size, "key")
    value_heads = self._linear_projection(value, self.value_size, "value")

    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)
    attn_logits = attn_logits / sqrt_key_size
    if mask is not None:
      if mask.ndim != attn_logits.ndim:
        raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                         f"{attn_logits.ndim}.")
      attn_logits = jnp.where(mask, attn_logits, -1e30)

    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    # Concatenate attention matrix of all heads into a single vector.
    attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
    if self.model_size is not None:
      return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)
    else:
      return attn_vec

  @hk.transparent
  def _linear_projection(
      self,
      x: jnp.ndarray,
      head_size: int,
      name: Optional[str] = None
  ) -> jnp.ndarray:
    y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
    return y.reshape((*x.shape[:-1], self.num_heads, head_size))

class RelationalLayer(base.Module):
  """Multi-head attention based relational layer."""

  def __init__(self,
    num_heads=2,
    w_init_scale=2.,
    init_bias=1.,
    position_embed=16,
    key_size=64,
    residual='skip',
    layernorm=True,
    shared_parameters=True, name='relational'):
    super().__init__(name=name)
    self.shared_parameters = shared_parameters
    self.num_heads = num_heads
    self.w_init_scale = w_init_scale
    self.position_embed = position_embed
    self.residual = residual
    self.key_size = key_size
    self.layernorm = layernorm
    self.init_bias = init_bias
    self.w_init = hk.initializers.VarianceScaling(w_init_scale)
    self.b_init = hk.initializers.Constant(init_bias)

  def __call__(self, factors: jnp.ndarray, queries: jnp.ndarray=None) -> jnp.ndarray:
    """Summary
    
    Args:
        factors (jnp.ndarray): B x N x D
        queries (jnp.ndarray, optional): B x N x D. If not given, factors are treated as queries.
    
    Returns:
        jnp.ndarray: Description
    """
    if self.layernorm:
      ln = hk.LayerNorm(axis=-1, param_axis=-1,
                  create_scale=True, create_offset=True)
      factors = ln(factors)

    if queries is None: 
      queries = factors
    else:
      if self.layernorm:
        queries = ln(queries)

    if self.shared_parameters:
      values = self.shared_attn(queries, factors)
      return self.residualfn(factors, values)
    else:
      values = self.independent_attn(queries, factors)
      return jax.vmap(self.residualfn)(factors, values)

  def residualfn(self, factors, values):
    D = factors.shape[-1]
    output = hk.Linear(D, w_init=self.w_init)(values)


    if self.residual == "skip":
      return factors + output

    elif self.residual == "concat":
      return jnp.concatenate((factors, output), axis=-1)

    elif self.residual == "output":
      x = factors
      y = output
      b = hk.get_parameter("b_gate", [D], x.dtype, init=self.b_init)
      return x + jax.nn.sigmoid(hk.Linear(D)(x) - b)*y

    elif self.residual == "sigtanh":
      x = factors
      y = output
      init = lambda size: self.init_bias*jnp.ones(size)
      b = hk.get_parameter("b_gate", [D], x.dtype, init=self.b_init)
      term1 = jax.nn.sigmoid(hk.Linear(D)(y) - b)
      term2 = jax.nn.tanh(hk.Linear(y))
      return x + term1*term2

    # elif self.residual == "gtrxl":
    #   e = factors
    #   y_bar = output
    #   y_hat = e + y_bar
    #   y = hk.LayerNorm(axis=-1, param_axis=-1,
    #               create_scale=True,
    #               create_offset=True)(y_hat)
    #   y = e + jax.nn.relu(y)
    #   e = 

    elif self.residual == "none":
      return output
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
    if self.position_embed:
      self.embedder = hk.Embed(
        vocab_size=N,
        embed_dim=self.position_embed)
      embeddings = self.embedder(jnp.arange(N)) # N x D
      concat = lambda a,b: jnp.concatenate((a, b), axis=-1)
      factors = jax.vmap(concat, in_axes=(1, None), out_axes=1)(
        factors, embeddings)


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
    queries_prep, factors_prep = self.prepare_data(queries, factors)
    D = factors.shape[-1]
    attn = MultiHeadAttention(
      num_heads=self.num_heads,
      key_size=self.key_size or D,
      model_size=None,
      w_init_scale=self.w_init_scale,
      )
    attn = jax.vmap(attn, in_axes=(0, None, None))

    out = attn(queries_prep, factors_prep, factors_prep)
    out = out.transpose((1,0,2))  # [B, N, D]
    return out

  def independent_attn(
      self,
      queries: jnp.ndarray,
      factors: jnp.ndarray,
      ) -> jnp.ndarray:
    raise NotImplementedError()