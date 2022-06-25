from typing import Sequence, Optional
from typing import NamedTuple
import jax
import jax.numpy as jnp
from acme.jax import networks as networks_lib

import functools
import haiku as hk
from utils import data as data_utils

from modules.basic_archs import AuxilliaryTask

from modules.embedding import embed_position
from modules.duelling import DuellingSfQNet
from utils import vmap
from utils import data as data_utils


from modules.usfa import UsfaHead, USFAPreds, USFAInputs, SfQNet

class FarmUsfaHead(UsfaHead):

  def __init__(self,
    cumulants_per_module: int,
    vmap_multihead: str = 'lift',
    relational_net = lambda x:x,
    position_embed: int=0,
    struct_policy: bool=False,
    argmax_mod: bool=False,

    **kwargs,
    ):
    super(FarmUsfaHead, self).__init__(
      state_dim=1, # will be ignored
      **kwargs)

    self.vmap_multihead = vmap_multihead
    self.relational_net = relational_net
    self._cumulants_per_module = cumulants_per_module
    self.sf_factory = lambda: hk.nets.MLP([self.hidden_size, self.num_actions*cumulants_per_module])
    self.policy_net_factory = lambda: hk.nets.MLP(
          [self.policy_size]*self.policy_layers)
    self.position_embed = position_embed
    self.struct_policy = struct_policy
    self.argmax_mod = argmax_mod


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


    # -----------------------
    # get input
    # -----------------------
    def concat(x,y): return jnp.concatenate((x,y), axis=-1)
    if self.struct_policy:
      # create 1-hot like vectors for each module with only its subset of task dimensions
      # e.g. [w1, w2, ...] --> [w1, 0, 0, 0]
      block_size = policy.shape[-1]//M
      ones = jnp.ones((1, block_size)).astype(policy.dtype)

      # Policy = [B, D_z]
      # [M, D_z]
      block_id = jax.scipy.linalg.block_diag(*[ones for _ in range(M)])

      mul = jax.vmap(jnp.multiply, in_axes=(None, 0))
      mul = jax.vmap(mul, in_axes=(0, None))
      # [B, M, W]
      struct_policy = mul(policy, block_id)
      policy_embed = hk.BatchApply(self.policynet)(struct_policy) # [B, N, D_z]
      state_policy = concat(state, policy_embed)

    else:
      policy_embed = self.policynet(policy) # [B, D_z]
      vmap_concat = jax.vmap(concat, in_axes=(1, None), out_axes=1)
      # [B, M, D_z+D_h]
      state_policy = vmap_concat(state, policy_embed)


    if self.layernorm == 'sf_input':
      state_policy = hk.LayerNorm(
          axis=-1,
          param_axis=-1,
          create_scale=False,
          create_offset=False)(state_policy)

    if self.layernorm == 'sf_input':
      state_policy = hk.LayerNorm(
          axis=-1,
          param_axis=-1,
          create_scale=False,
          create_offset=False)(state_policy)

    if self.multihead:
      # [B, M, A*C]
      sf = vmap.batch_multihead(
        fn=self.sf_factory,
        x=state_policy,
        vmap=self.vmap_multihead)
    else:
      # [B, M, A*C]
      if self.position_embed:
        state_policy = embed_position(
          factors=state_policy, size=self.position_embed)
      sf = hk.BatchApply(self.sf_factory())(state_policy)

    sf = jnp.reshape(sf, [B, M, A, C])
    # [B, A, C*M=D_w]
    sf = sf.transpose(0, 2, 1, 3).reshape(B, A, -1)

    if self.layernorm == 'sf':
      sf = hk.LayerNorm(
          axis=-1,
          param_axis=-1,
          create_scale=False,
          create_offset=False)(sf)

    # vmap loop over A for SF
    q_prod = jax.vmap(jnp.multiply, in_axes=(1, None), out_axes=1)(sf, task)
    

    return sf, q_prod

  def sfgpi(self,
    inputs: USFAInputs,
    z: jnp.ndarray,
    w: jnp.ndarray,
    key: networks_lib.PRNGKey,
    setting='train',
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

    compute_sf = jax.vmap(self.compute_sf,
      in_axes=(None, 1, None), out_axes=1)

    memory_out = self.relational_net(inputs.memory_out)
    # [B, N, A, D_w]
    sf, q_values_prod = compute_sf(memory_out, z, w)

    if self.argmax_mod and setting=='eval':
      M = memory_out.shape[1]
      # [B, N, A, M, D_w/M]
      q_values_mod = jnp.stack(jnp.split(q_values_prod, M, axis=-1), axis=2)
      # [B, N, A, M]
      q_values = q_values_mod.sum(-1)
      # -----------------------
      # GPI modules
      # -----------------------
      # [B, N, A, M] --> [B, N, A]
      q_values = jnp.max(q_values, axis=-1)
    else:
      q_values = q_values_prod.sum(-1)


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
