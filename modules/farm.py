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
  r"""Long short-term memory (LSTM) RNN core.
  The implementation is based on :cite:`zaremba2014recurrent`. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
  computes
  .. math::
     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}
  where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
  output gate activations, and :math:`g_t` is a vector of cell updates.
  The output is equal to the new hidden, :math:`h_t`.
  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add 1.0 to :math:`b_f`
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.
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

    gated = vmap.vmap_multihead(
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
    state = LSTMState(hidden=jnp.zeros([self.hidden_size]),
                      cell=jnp.zeros([self.hidden_size]))
    state = add_batch(state, self.nmodules)
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class FeatureAttention(hk.Module):
  """FeatureAttention from Feature-Attending Recurrent Modules. f_att from paper.
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
    coefficients = vmap.vmap_multihead(
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



class FARM(hk.RNNCore):
  def __init__(self,
    module_size: int,
    nmodules: int,
    projection_dim: int=16,
    name: Optional[str] = None):
    """
    """
    super().__init__(name=name)
    self.memory = StructuredLSTM(module_size, nmodules)
    self.obs_attention = FeatureAttention(projection_dim)

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

    image_attn = self.image_attention(query, inputs.image)

    # module_attn = self.module_attention(query, prev_state.hidden)


    memory_input = jnp.concatenate((query, image_attn), axis=-1)
    # memory_input = jnp.concatenate((query, image_attn, module_attn))
    hidden, state = self.memory(memory_input, prev_state)

    return hidden, state

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    return self.memory.initial_state(batch_size)


  def image_attention(self, query, image):
    """Apply attention and flatten output"""
    attn_out = self.obs_attention(query, image)
    B, N = attn_out.shape[:2]
    return attn_out.reshape(B, N, -1)

  def module_attention(self, query, image):
    return []


class FarmSharedOutput(FARM):
  """docstring for FarmSharedOutput"""
  def __init__(self, out_layers, *args, **kwargs):
    super(FarmSharedOutput, self).__init__(*args, **kwargs)
    assert out_layers >=0
    if out_layers == 0:
      self.out_mlp = lambda x:x
    else:
      self.out_mlp = hk.nets.MLP([self.module_size]*out_layers)

  def __call__(self, *args, **kwargs) -> Tuple[jnp.ndarray, LSTMState]:
    hidden, state = super().__call__(*args, **kwargs)
    hidden = hk.BatchApply(self.out_mlp)(hidden)
    return hidden, state

