from typing import NamedTuple, Optional, Tuple
import haiku as hk
import jax
import jax.numpy as jnp

from utils import vmap

def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)


def expand_tile_dim(x, dim, size):
  """E.g. shape=[1,128] --> [1,10,128] if dim=1, size=10
  """
  ndims = len(x.shape)
  x = jnp.expand_dims(x, dim)
  tiling = [1]*dim + [size] + [1]*(ndims-dim)
  return jnp.tile(x, tiling)


class LSTMState(NamedTuple):
  """An LSTM core state consists of hidden and cell vectors.
  Attributes:
    hidden: Hidden state.
    cell: Cell state.
  """
  hidden: jnp.ndarray
  cell: jnp.ndarray

class FarmInputs(NamedTuple):
  """An LSTM core state consists of hidden and cell vectors.
  Attributes:
    hidden: Hidden state.
    cell: Cell state.
  """
  image: jnp.ndarray
  vector: jnp.ndarray

class StructuredLSTM(hk.RNNCore):
  r""" Structured long short-term memory (LSTM) RNN core.
  This acts as N indepedently updating RNNs.
  """

  def __init__(self, hidden_size: int, nmodules: int, name: Optional[str] = None):
    """Constructs an LSTM.
    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.nmodules = nmodules

  def __call__(
      self,
      inputs: jnp.ndarray, # [B, N, D]
      prev_state: LSTMState,
  ) -> Tuple[jnp.ndarray, LSTMState]:

    if len(inputs.shape) == 0 or \
       len(inputs.shape) > 3 or \
       not inputs.shape:
      raise ValueError("LSTM input must be rank-2 or rank-3.")

    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)

    gated = vmap.batch_multihead(
      fn=lambda: hk.Linear(4 * self.hidden_size),
      x=x_and_h
    )

    # TODO(slebedev): Consider aligning the order of gates with Sonnet.
    # i = input, g = cell_gate, f = forget_gate, o = output_gate
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)

    return h, LSTMState(h, c)

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    state = LSTMState(hidden=jnp.zeros([self.nmodules, self.hidden_size]),
                      cell=jnp.zeros([self.nmodules, self.hidden_size]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state

class FeatureAttention(hk.Module):
  """FeatureAttention from Feature-Attending Recurrent Modules. f_att from paper.
  Each module has its own attention parameters.
  """
  def __init__(self, dim=16):
    super(FeatureAttention, self).__init__()
    self.dim = dim

  def __call__(
      self,
      queries: jnp.ndarray, # [B, N, D]
      image: jnp.ndarray, # [B, N, H, W, C]
  ) -> jnp.ndarray:

    B, N, D = queries.shape
    if len(image.shape) == 4:
      image = expand_tile_dim(image, dim=1, size=N)
    elif len(image.shape) == 5: # already expanded
      pass
    else:
      raise ValueError("image input must be rank-5 or rank-6.")

    # ======================================================
    # compute coefficients
    # ======================================================
    # function will create N copies
    coefficients = vmap.batch_multihead(
      fn=lambda: hk.Linear(self.dim),
      x=queries)
    coefficients = jnp.expand_dims(coefficients, (2,3)) # [B, N, H, W, D]
    coefficients = jax.nn.sigmoid(coefficients)

    # ======================================================
    # attend + projections
    # ======================================================
    # first projection
    image = hk.BatchApply(hk.Conv2D(self.dim, [1, 1], 1))(image)

    # attend
    image = image*coefficients

    # second projection
    image = hk.BatchApply(hk.Conv2D(self.dim, [1, 1], 1))(image)

    return image

class ModuleAttention(hk.Module):
  """Attention over modules using transformer-style attention. 
  Each module has its own parameters. f_share from paper."""
  def __init__(self, module_size, num_heads=4, w_init_scale=2.,
    shared_parameters=True):
    super(ModuleAttention, self).__init__()
    self.attn_factory = lambda: hk.MultiHeadAttention(
      num_heads=num_heads,
      key_size=module_size,
      model_size=module_size,
      w_init_scale=w_init_scale,
      )
    self.shared_parameters = shared_parameters

  def prepare_data(
      self,
      queries: jnp.ndarray,
      hidden_states: jnp.ndarray,
      ) -> jnp.ndarray:
    """ Multihead attention expects [N, B, D]. Fix. """
    # -----------------------
    # prepare data
    # -----------------------
    # convert things to dims expected by multihead attention
    hidden_states = hidden_states.transpose((1,0,2))  # [N, B, D]
    queries = queries.transpose((1,0,2))  # [N, B, D]

    N, B = queries.shape[:2]
    D = hidden_states.shape[2]

    # add zeros for no attention
    zeros = jnp.zeros((1, B, D))
    hidden_states = jnp.concatenate((hidden_states, zeros)) # [N+1, B, D]

    return queries, hidden_states

  def __call__(
      self,
      queries: jnp.ndarray,
      hidden_states: jnp.ndarray,
      ) -> jnp.ndarray:
    """
    Args:
        queries (jnp.ndarray): B x N x D
        hidden_states (jnp.ndarray): B x N x D
    
    Returns:
        jnp.ndarray: Description
    """
    if self.shared_parameters:
      return self.shared_attn(queries, hidden_states)
    else:
      return self.independent_attn(queries, hidden_states)

  def shared_attn(
      self,
      queries: jnp.ndarray,
      hidden_states: jnp.ndarray,
      ) -> jnp.ndarray:
    """Share parameters across queries.
    
    Args:
        queries (jnp.ndarray): [B, N, D]
        hidden_states (jnp.ndarray): [B, N, D]
    
    Returns:
        jnp.ndarray: Description
    """
    queries, hidden_states = self.prepare_data(queries, hidden_states)

    attn = self.attn_factory()
    attn = jax.vmap(attn, in_axes=(0, None, None))

    out = attn(queries, hidden_states, hidden_states)
    out = out.transpose((1,0,2))  # [B, N, D]
    return out

  def independent_attn(
      self,
      queries: jnp.ndarray,
      hidden_states: jnp.ndarray,
      ) -> jnp.ndarray:
    """Each query gets indepedent set of parameters
    
    Args:
        queries (jnp.ndarray): [B, N, D]
        hidden_states (jnp.ndarray): [B, N, D]
    
    Returns:
        jnp.ndarray: Description
    """
    queries, hidden_states = self.prepare_data(queries, hidden_states)

    # -----------------------
    # make functions
    # -----------------------
    def apply_attn(x):
      return self.attn_factory()(x[0], x[1], x[2])
    functions = [apply_attn for i in jnp.arange(N)]

    # -----------------------
    # initialization:
    #  just make N copies of function
    # -----------------------
    if hk.running_init():
      # during initialization, just create functions
      q = queries[0]  # [1, B, D]
      h = hidden_states  # [N+1, B, D]
      # reuse same example for all functions
      x = [f((q, h, h)) for f in functions]

      # combine all outputs at leaf jnp.ndarray level
      out = jax.tree_map(lambda *arrays: jnp.stack(arrays), *x)

    # -----------------------
    # apply:
    #  vmap magic
    # -----------------------
    else:
      index = jnp.arange(N)
      # during apply, apply functions in parallel
      vmap_functions = hk.vmap(
        # switch can only take 1 input so make tuple
        lambda i, q, k, v: hk.switch(i, functions, (q, k, v)),
        # select by {0th, 1st} for {index, queries}, and make copies
        #   for hidden_states
        # doing this will mimic sizes used during initialization
        in_axes=(0, 0, None, None),
        split_rng=True)
      out = vmap_functions(index, queries, hidden_states, hidden_states)

    # remove dummy dimension & return to batch-first
    out = out[:, 0] # [N, B, D]
    out = out.transpose((1,0,2))  # [B, N, D]
    return out

class FARM(hk.RNNCore):
  def __init__(self,
    module_size: int,
    nmodules: int,
    attn_size: int = None,
    nattn_heads: int=4,
    shared_attn_params: bool=True,
    projection_dim: int=16,
    name: Optional[str] = None):
    """
    """
    super().__init__(name=name)
    self.memory = StructuredLSTM(module_size, nmodules)
    self._feature_attention = FeatureAttention(projection_dim)
    self._module_attention = ModuleAttention(
      module_size=attn_size or module_size,
      num_heads=nattn_heads,
      shared_parameters=shared_attn_params)

    self.module_size = module_size
    self.nmodules = nmodules
    self.projection_dim = projection_dim

  def __call__(
      self,
      inputs: FarmInputs,
      prev_state: LSTMState, # [B, N, D]
  ) -> Tuple[jnp.ndarray, LSTMState]:

    vector_tiled = expand_tile_dim(inputs.vector, dim=1,
      size=self.nmodules)
    query = jnp.concatenate((prev_state.hidden, vector_tiled),
      axis=-1)

    image_attn = self.image_attention(query, inputs.image)  # [B, N , D]
    module_attn = self.module_attention(query, prev_state.hidden)  # [B, N , D]

    memory_input = jnp.concatenate((query, image_attn, module_attn), axis=-1)
    hidden, state = self.memory(memory_input, prev_state)

    return hidden, state

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    return self.memory.initial_state(batch_size)


  def image_attention(self, query, image):
    """Apply attention and flatten output"""
    attn_out = self._feature_attention(query, image)
    B, N = attn_out.shape[:2]
    return attn_out.reshape(B, N, -1)

  def module_attention(self, query, hidden_states):
    return self._module_attention(query, hidden_states)


  @property
  def total_dim(self):
    return self.module_size*self.nmodules

class FarmSharedOutput(FARM):
  """docstring for FarmSharedOutput"""
  def __init__(self, out_layers, *args, **kwargs):
    super(FarmSharedOutput, self).__init__(*args, **kwargs)
    assert out_layers >=0
    if out_layers == 0:
      self.out_mlp = lambda x:x
      raise RuntimeError
    else:
      self.out_mlp = hk.nets.MLP([self.module_size]*out_layers)

  def __call__(self, *args, **kwargs) -> Tuple[jnp.ndarray, LSTMState]:
    hidden, state = super().__call__(*args, **kwargs)
    new_hidden = hk.BatchApply(self.out_mlp)(hidden)
    return new_hidden, state

