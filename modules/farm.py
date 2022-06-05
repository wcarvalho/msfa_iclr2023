from typing import NamedTuple, Optional, Tuple
import haiku as hk
import jax
import jax.numpy as jnp

from utils import vmap
from utils import data as data_utils
from modules.relational import RelationalLayer

def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)


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

class FarmOutputs(NamedTuple):
  """An LSTM core state consists of hidden and cell vectors.
  Attributes:
    hidden: Hidden state.
    cell: Cell state.
  """
  hidden: jnp.ndarray
  attn: jnp.ndarray

class StructuredLSTM(hk.RNNCore):
  """ Structured long short-term memory (LSTM) RNN core.
  This acts as N indepedently updating RNNs.
  """

  def __init__(self, hidden_size: int, nmodules: int, vmap: str = 'switch',name: Optional[str] = None):
    """Constructs an LSTM.
    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.nmodules = nmodules
    self.vmap = vmap

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
      x=x_and_h,
      vmap=self.vmap,
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
  def __init__(self, dim=16, normalize=False, vmap: str="switch"):
    super(FeatureAttention, self).__init__()
    self.dim = dim
    self.vmap = vmap
    self.normalize = normalize

  def __call__(
      self,
      queries: jnp.ndarray, # [B, N, D]
      image: jnp.ndarray, # [B, H, W, C]
  ) -> jnp.ndarray:

    B, N, D = queries.shape
    # ======================================================
    # compute coefficients
    # ======================================================
    # function will create N copies
    coefficients = vmap.batch_multihead(
      fn=lambda: hk.Linear(self.dim),
      x=queries,
      vmap=self.vmap)
    # [B, N, D]
    coefficients = jax.nn.sigmoid(coefficients)

    if self.normalize:
      # encourages competition
      coefficients = coefficients/(1e-5 + jnp.sum(coefficients, axis=1, keepdims=1))

    # [B, N, H, W, D]
    coefficients = jnp.expand_dims(coefficients, (2,3)) 


    # ======================================================
    # attend + projections
    # ======================================================
    # first projection
    image = hk.BatchApply(hk.Conv2D(self.dim, [1, 1], 1))(image)

    # [B, H, W, C], [B, N, C]
    multiply = jax.vmap(jnp.multiply, in_axes=(None, 1), out_axes=1)
    image = multiply(image, coefficients)

    # second projection
    image = hk.BatchApply(hk.Conv2D(self.dim, [1, 1], 1))(image)

    return image

def get_farm_sizes(module_size, nmodules, memory_size):
  isnone = [x is None for x in [module_size, nmodules, memory_size]]
  assert sum(isnone) <=1, "underspecified"

  if module_size is None:
    module_size = memory_size / nmodules
  elif nmodules is None:
    nmodules = memory_size/module_size
  elif memory_size is None:
    memory_size = nmodules*module_size

  assert memory_size == nmodules *module_size

  return int(module_size), int(nmodules), int(memory_size)


class FARM(hk.RNNCore):
  def __init__(self,
    module_size: int,
    nmodules: int,
    memory_size: int=None,
    module_attn_size: int = None,
    module_attn_heads: int=4,
    shared_module_attn: bool=True,
    projection_dim: int=16,
    image_attn: bool=True,
    return_attn: bool=False,
    normalize_attn: bool=False,
    vmap: str = 'switch',
    name: Optional[str] = None):
    """
    Args:
        module_size (int): Description
        nmodules (int): Description
        module_attn_size (int, optional): Description
        module_attn_heads (int, optional): Description
        shared_module_attn (bool, optional): Description
        projection_dim (int, optional): Description
        vmap (bool, optional): whether to vmap over modules or use for loops. currently for loops faster... need to investigate...
        name (Optional[str], optional): Description
    """
    super().__init__(name=name)

    module_size, nmodules, memory_size = get_farm_sizes(module_size, nmodules, memory_size)

    if module_attn_heads > 0 and module_attn_heads < 1:
      module_attn_heads = int(nmodules*float(module_attn_heads))
    self.module_attn_heads = module_attn_heads

    self.memory = StructuredLSTM(module_size, nmodules, vmap=vmap)
    self._feature_attention = FeatureAttention(
      projection_dim, 
      normalize=normalize_attn,
      vmap=vmap)
    self.image_attn = image_attn

    if module_attn_heads > 0:
      if module_attn_size:
        attn_size=module_attn_size*module_attn_heads
      else:
        attn_size=module_size*module_attn_heads
      self._module_attention = RelationalLayer(
        num_heads=module_attn_heads, 
        shared_parameters=shared_module_attn,
        attn_size=attn_size
        )

    self.module_size = module_size
    self.nmodules = nmodules
    self.memory_size = memory_size
    self.projection_dim = projection_dim
    self.return_attn = return_attn

  def __call__(
      self,
      inputs: FarmInputs,
      prev_state: LSTMState, # [B, N, D]
      ) -> Tuple[jnp.ndarray, LSTMState]:

    def concat(x,y):
      return jnp.concatenate((x, y), axis=-1)

    # vmap dim N of hidden, copy inputs.vector N times
    query = jax.vmap(concat, in_axes=(1, None), out_axes=1)(
        # [B, N, D]        [B, D]
        prev_state.hidden, inputs.vector)

    # [B, N, H, W, D]
    image_attn = self.image_attention(query, inputs.image)
    B, N = image_attn.shape[:2]
    image_attn_flat = image_attn.reshape(B, N, -1)

    if self.module_attn_heads > 0:
      # [B, N , D]
      query = self.module_attention(query, prev_state.hidden)

    memory_input = [query, image_attn_flat]
    memory_input = jnp.concatenate(memory_input, axis=-1)
    hidden, state = self.memory(memory_input, prev_state)

    if self.return_attn:
      output = FarmOutputs(hidden=hidden, attn=image_attn)
    else:
      output = hidden

    return output, state

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    return self.memory.initial_state(batch_size)


  def image_attention(self, query, image):
    """Apply attention and flatten output"""
    if self.image_attn:
      return self._feature_attention(query, image)
    else:
      B, N = query.shape[:2]
      # [B, H, W, C]
      attn_out = data_utils.expand_tile_dim(image, axis=1, size=N)
      return attn_out

  def module_attention(self, query, hidden_states):
    return self._module_attention(queries=query, factors=hidden_states)


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
    else:
      self.out_mlp = hk.nets.MLP([self.module_size]*out_layers)

  def __call__(self, *args, **kwargs) -> Tuple[jnp.ndarray, LSTMState]:
    hidden, state = super().__call__(*args, **kwargs)
    new_hidden = hk.BatchApply(self.out_mlp)(hidden)
    return new_hidden, state

