from typing import Sequence, Optional
from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from acme.jax.networks import duelling

import functools
import haiku as hk
from utils import data as data_utils

from modules.basic_archs import AuxilliaryTask
from modules.embedding import embed_position
from modules.duelling import DuellingSfQNet
from modules.usfa import SfQNet
from utils import vmap
from utils import data as data_utils

class FarmUvfaInputs(NamedTuple):
  w: jnp.ndarray  # task vector
  memory_out: jnp.ndarray  # memory output (e.g. LSTM)

class FarmUvfaOutputs(NamedTuple):
  q: jnp.ndarray  # chosen q-value


class FarmUvfaHead(hk.Module):
  """docstring for FarmUvfa"""
  def __init__(self, 
    num_actions: int,
    hidden_sizes : Sequence[int]=[128],
    task_embed_dim : int=32,
    task_embed_layers : int=2,
    struct_w_input: bool=False,
    dot_heads: bool=False):
    """Summary
    
    Args:
        num_actions (int): Description
        hidden_size (int, optional): Description
        task_embed_dim (int, optional): Description
        task_embed_layers (int, optional): Description
        struct_w_input (bool, optional): break up task vector evenly among modules
    """
    super(FarmUvfaHead, self).__init__()

    self.num_actions = num_actions
    self.hidden_sizes = hidden_sizes
    self.struct_w_input = struct_w_input
    self.dot_heads = dot_heads

    if task_embed_layers > 0:
      self.tasknet = hk.nets.MLP(
          [task_embed_dim]*task_embed_layers)
    else:
      self.tasknet = lambda x:x



  def __call__(self, inputs : FarmUvfaInputs, **kwargs):

    # -----------------------
    # inputs
    # -----------------------
    task = inputs.w
    hidden = inputs.memory_out.hidden
    nmodules = hidden.shape[1]
    task_dim = task.shape[-1]
    task_embed = self.tasknet(task)

    # -----------------------
    # get q-input
    # -----------------------
    concat = lambda x, y: jnp.concatenate((x,y), axis=-1)

    if self.struct_w_input:
      # [B, M/ W/M]
      struct_task = jnp.stack(jnp.split(task_embed, nmodules, axis=-1), axis=1)
      q_input = concat(hidden, struct_task)
    else:
      vmap_concat = jax.vmap(concat, in_axes=(1, None), out_axes=1)
      q_input = vmap_concat(hidden, task_embed)


    if self.dot_heads:
      C = num_cumulants = task_dim//nmodules
      A = self.num_actions
      M = nmodules
      B = hidden.shape[0]
      q_net = hk.nets.MLP(self.hidden_sizes+[self.num_actions*num_cumulants])

      # [B, M, A*C]
      q_tilde = hk.BatchApply(q_net)(q_input)
      # [B, M, A, C]
      q_tilde = jnp.reshape(q_tilde, [B, M, A, C])
      # [B, A, M*C]
      q_tilde = q_tilde.transpose(0, 2, 1, 3).reshape(B, A, -1)
      q_vals = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)(q_tilde, task)

      # [B, A]
      q_vals = q_vals.sum(-1)
    else:
      q_net = duelling.DuellingMLP(
        num_actions=self.num_actions,
        hidden_sizes=self.hidden_sizes)
      # [B, M, Q]
      q_vals = hk.BatchApply(q_net)(q_input)
      q_vals = q_vals.sum(1)


    return FarmUvfaOutputs(q=q_vals)