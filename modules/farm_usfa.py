from typing import Sequence, Optional
from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib

import functools
import haiku as hk
from utils import data as data_utils

from modules.basic_archs import AuxilliaryTask
from modules.embedding import OneHotTask
from modules.duelling import DuellingSfQNet
from utils import vmap
from utils import data as data_utils


from modules.usfa import UsfaHead, USFAPreds, USFAInputs, SfQNet

class FarmUsfaHead(UsfaHead):

  def __init__(self,
    cumulants_per_module: int,
    vmap_multihead: str = 'lift',
    relational_net = lambda x:x,
    **kwargs,
    ):
    super(FarmUsfaHead, self).__init__(
      state_dim=1, # will be ignored
      **kwargs)

    self.vmap_multihead = vmap_multihead
    self.relational_net = relational_net
    self._cumulants_per_module = cumulants_per_module
    self.sf_factory = lambda: hk.nets.MLP([self.hidden_size, self.num_actions*cumulants_per_module])

  def compute_sf(self,
    state : jnp.ndarray,
    policy : jnp.ndarray,
    task : jnp.ndarray):
    """Summary
    
    Args:
        state (jnp.ndarray): B x M x D_h
        policy (jnp.ndarray): B x D_z
        task (jnp.ndarray): B x D_w
    """
    B, M, _ = state.shape
    A, C = self.num_actions, self._cumulants_per_module

    def concat(x,y): return jnp.concatenate((x,y), axis=-1)
    concat = jax.vmap(concat, in_axes=(1, None), out_axes=1)
    # [B, M, D_z+D_h]
    state_policy = concat(state, policy)

    if self.multihead:
      # [B, M, A*C]
      sf = vmap.batch_multihead(
        fn=self.sf_factory,
        x=state_policy,
        vmap=self.vmap_multihead)
    else:
      # [B, M, A*C]
      sf = hk.BatchApply(self.sf_factory())(state_policy)

    sf = jnp.reshape(sf, [B, M, A, C])
    # [B, A, C*M=D_w]
    sf = sf.transpose(0, 2, 1, 3).reshape(B, A, -1)

    q = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)(sf, task)
    q = q.sum(-1)

    return sf, q

  def sfgpi(self,
    inputs: USFAInputs,
    z: jnp.ndarray,
    w: jnp.ndarray,
    key: networks_lib.PRNGKey,
    **kwargs) -> USFAPreds:
    """M = number of modules. N = number of policies.

    Args:
        inputs (USFAInputs): Description
        z (jnp.ndarray): B x N x D_z
        w (jnp.ndarray): B x D_w
        key (networks_lib.PRNGKey): Description
    
    Returns:
        USFAPreds: Description
    """

    z_embedding = hk.BatchApply(self.policynet)(z) # [B, N, D_z]

    compute_sf = jax.vmap(self.compute_sf,
      in_axes=(None, 1, None), out_axes=1)

    # [B, N, A, D_w], [B, N, A]
    memory_out = self.relational_net(inputs.memory_out)
    sf, q_values = compute_sf(memory_out, z_embedding, w)

    # -----------------------
    # GPI
    # -----------------------
    # [B, N, A] --> [B, A]
    q_values = jnp.max(q_values, axis=1)

    # -----------------------
    # prepare other vectors
    # -----------------------
    # [B, N, D_z] --> # [B, N, A, D_z]
    z = data_utils.expand_tile_dim(z, axis=2, size=self.num_actions)

    return USFAPreds(
      sf=sf,       # [B, N, A, D_w]
      z=z,         # [B, N, A, D_w]
      q=q_values,  # [B, N, A]
      w=w)         # [B, D_w]


  @property
  def out_dim(self):
    return self.sf_out_dim

  @property
  def cumulants_per_module(self):
    return self._cumulants_per_module
