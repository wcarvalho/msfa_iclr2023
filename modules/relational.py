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

class GruGate(hk.Module):
  """docstring for GruGate"""
  def __init__(self, hidden_size, b_init=None, w_init=None):
    super(GruGate, self).__init__()
    self.b_init = b_init or jnp.zeros
    self.w_init = w_init or hk.initializers.VarianceScaling()
    self.hidden_size = hidden_size

  def __call__(self, queries, values):
    input_size = values.shape[-1]
    hidden_size = self.hidden_size
    w_i = hk.get_parameter("w_i", [input_size, 3 * hidden_size], values.dtype,
                           init=self.w_init)
    w_h = hk.get_parameter("w_h", [hidden_size, 3 * hidden_size], values.dtype,
                           init=self.w_init)
    b = hk.get_parameter("b", [3 * hidden_size], values.dtype, init=self.b_init)
    w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[2 * hidden_size], axis=1)
    b_z, b_a = jnp.split(b, indices_or_sections=[2 * hidden_size], axis=0)

    gates_x = jnp.matmul(values, w_i)
    zr_x, a_x = jnp.split(
        gates_x, indices_or_sections=[2 * hidden_size], axis=-1)
    zr_h = jnp.matmul(queries, w_h_z)
    zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
    z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

    a_h = jnp.matmul(r * queries, w_h_a)
    a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

    next_queries = (1 - z) * queries + z * a
    return next_queries

class Gate(hk.Module):
  """docstring for Gate"""
  def __init__(self,
    w_init=2.0,
    init_bias=2.0,
    residual='sigtanh',
    relu_gate=False,
    layernorm=False):
    super(Gate, self).__init__()
    self.w_init = hk.initializers.VarianceScaling(w_init)
    self.b_init = hk.initializers.Constant(init_bias)
    self.relu_gate = relu_gate
    self.residual = residual
    self.layernorm =layernorm

  def __call__(self, queries, values):
    D = queries.shape[-1]
    linear = lambda x: hk.Linear(D, with_bias=False, w_init=self.w_init)(x)

    if self.relu_gate:
      output = jax.nn.relu(linear(values))

    if self.residual == "skip":
      output = queries + linear(values)

    elif self.residual == "concat":
      output = jnp.concatenate((queries, linear(values)), axis=-1)

    elif self.residual == "output":
      b = hk.get_parameter("b_gate", [D], queries.dtype, init=self.b_init)
      gate = jax.nn.sigmoid(linear(queries) - b)
      output = queries + gate*linear(values)

    elif self.residual == "sigtanh":
      b = hk.get_parameter("b_gate", [D], queries.dtype, init=self.b_init)
      gate = jax.nn.sigmoid(linear(values) - b)
      output = jax.nn.tanh(linear(values))
      output = queries + gate*output

    elif self.residual == "gru":
      gate = GruGate(hidden_size=D, w_init=self.w_init, b_init=self.b_init)
      output = gate(queries=queries, values=output)

    elif self.residual == "none":
      output = linear(values)
    else:
      raise NotImplementedError

    if self.layernorm:
      output = hk.LayerNorm(
        axis=-1,
        param_axis=-1,
        create_scale=True,
        create_offset=True)(output)

    return output

class RelationalLayer(base.Module):
  """Multi-head attention based relational layer."""

  def __init__(self,
    num_heads=2,
    w_init_scale=2.,
    res_w_init_scale=.2,
    init_bias=1.,
    position_embed=0,
    attn_size=256,
    residual='sigtanh',
    layernorm=False,
    relu_gate=False,
    pos_mlp=False,
    # add_zeros=False,
    shared_parameters=True, name='relational'):
    super().__init__(name=name)
    self.shared_parameters = shared_parameters
    self.num_heads = num_heads
    self.w_init_scale = w_init_scale
    self.position_embed = position_embed
    self.residual = residual
    self.attn_size = attn_size
    self.key_size = attn_size // num_heads
    self.layernorm = layernorm
    self.init_bias = init_bias
    self.relu_gate = relu_gate
    self.pos_mlp = pos_mlp
    # self.w_init = hk.initializers.VarianceScaling(w_init_scale)
    # self.res_w_init = hk.initializers.VarianceScaling(res_w_init_scale)
    # self.b_init = hk.initializers.Constant(init_bias)

    self.attn_gate = Gate(
      w_init=res_w_init_scale,
      init_bias=init_bias,
      residual=residual,
      relu_gate=relu_gate,
      layernorm=layernorm)
    self.mlp_gate = Gate(
      w_init=res_w_init_scale,
      init_bias=init_bias,
      residual=residual,
      relu_gate=relu_gate,
      layernorm=layernorm)

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
      values = self.attn_gate(queries, values)

      if self.pos_mlp:
        mlp = hk.nets.MLP(
          output_sizes=[values.shape[-1]]*2,
          activate_final=self.relu_gate)
        values_mlp = mlp(values)
        values = self.mlp_gate(values, values_mlp)

      return values


    else:
      values = self.independent_attn(queries, factors)
      raise NotImplementedError
      return jax.vmap(self.residualfn)(queries, values)

  def prepare_data(
      self,
      queries: jnp.ndarray,
      factors: jnp.ndarray,
      ) -> jnp.ndarray:
    """ """
    # -----------------------
    # prepare data
    # -----------------------
    # convert things to dims expected by multihead attention
    # factors = factors.transpose((1,0,2))  # [N, B, D]
    # queries = queries.transpose((1,0,2))  # [N, B, D]

    B, N = queries.shape[:2]
    if self.position_embed > 0:
      self.embedder = hk.Embed(
        vocab_size=N,
        embed_dim=self.position_embed)
      embeddings = self.embedder(jnp.arange(N)) # N x D
      concat = lambda a,b: jnp.concatenate((a, b), axis=-1)
      factors = jax.vmap(concat, in_axes=(0, None), out_axes=0)(
        factors, embeddings)

    D = factors.shape[2]
    # add zeros for no attention
    zeros = jnp.zeros((B, 1, D))
    factors = jnp.concatenate((factors, zeros), axis=1) # [B, N+1, D]

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

    attn_vmap = jax.vmap(attn)
    # B, N, D
    out = attn_vmap(queries_prep, factors_prep, factors_prep)

    return out

  def independent_attn(
      self,
      queries: jnp.ndarray,
      factors: jnp.ndarray,
      ) -> jnp.ndarray:
    raise NotImplementedError()

class RelationalNet(base.Module):
  """docstring for RelationalNet"""
  def __init__(self, *args, layers=1, **kwargs):
    super(RelationalNet, self).__init__()

    self.layers = [RelationalLayer(*args, **kwargs) for _ in range(layers)]

  def __call__(self, factors: jnp.ndarray) -> jnp.ndarray:

    for idx, layer in enumerate(self.layers):
      factors = layer(factors)

    return factors
