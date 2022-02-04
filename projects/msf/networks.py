import dataclasses

from typing import Callable, Optional, Tuple, NamedTuple

from acme import types
from acme.jax import networks as networks_lib
from acme.jax.networks import base
from acme.jax.networks import embedding
from acme.jax.networks import duelling
from acme.wrappers import observation_action_reward
import functools

import haiku as hk
import jax
import jax.numpy as jnp

from agents.td_agent.types import Predictions

from modules.farm import StructuredLSTM, FARM, FarmInputs


# ======================================================
# Types
# ======================================================
Images = jnp.ndarray

class USFAPreds(NamedTuple):
  q: jnp.ndarray
  sf: jnp.ndarray
  policy_zeds: jnp.ndarray


class USFAUnsupPreds(NamedTuple):
  q: jnp.ndarray
  sf: jnp.ndarray
  policy_zeds: jnp.ndarray
  cumulants: jnp.ndarray

# ======================================================
# small utils
# ======================================================
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

def process_inputs(inputs):
  last_action = inputs.action
  last_reward = inputs.reward
  observation = inputs.observation


  dtype = last_reward.dtype
  image = observation.image.astype(dtype)/255.0
  task = observation.task.astype(dtype)
  state_feat = observation.state_features.astype(dtype)


  last_action = jnp.expand_dims(last_action, -1)
  last_reward = jnp.expand_dims(last_reward, -1)

  return image, task, state_feat, last_action, last_reward


# ======================================================
# USFA
# ======================================================

def sample_gauss(mean, var, key, nsamples, samples_axis):
  # gaussian (mean=mean, var=.1I)
  samples = expand_tile_dim(mean, dim=samples_axis, size=nsamples)
  dims = samples.shape # [?, B, N, D]
  samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
  return samples.astype(mean.dtype)


def usfa(memory_out,
        task,
        statefn,
        policyfn,
        successorfn,
        num_actions,
        nsamples,
        state_dim,
        var,
        key,
        nbatchdims):
  """
  1. Sample K policy embeddings according to state features
  2. Compute corresponding successor features
  3. compute policy and do GPI
  
  Args:
      memory_out (TYPE): e.g. LSTM hidden
      task (TYPE): task vectors
      statefn (TYPE): net for memory to get state rep
      policyfn (TYPE): net for task to get policy rep
      successorfn (TYPE): net for [state, policy]
      num_actions (TYPE): number of actions
      nsamples (TYPE): number of policies to sample
      state_dim (TYPE): state dim of cumulant
      var (TYPE): variance for policy sampling
      key (TYPE): rng key
      nbatchdims (TYPE): number of dimensions before data. E.g. [T,B]=2
  
  Returns:
      TYPE: Description
  """
  dtype = memory_out.dtype
  batchdims = [1]*nbatchdims
  policy_axis = nbatchdims

  if nbatchdims == 1:
    batch_apply = lambda x:x
  else:
    batch_apply = hk.BatchApply
  num_dims = nbatchdims + 1 # for policy axis


  state = batch_apply(statefn)(memory_out)
  # -----------------------
  # add dim for K=nsamples of policy params
  # tile along that dimension
  # -----------------------
  state = jnp.expand_dims(state, axis=policy_axis)
  task_sampling = jnp.expand_dims(task, axis=policy_axis)
  # add one extra for action
  policy_zeds = jnp.expand_dims(task, axis=(policy_axis,policy_axis+1))
  state = jnp.tile(state, [*batchdims,nsamples,1])
  task_sampling = jnp.tile(task_sampling, [*batchdims,nsamples,1])
  policy_zeds = jnp.tile(policy_zeds, [*batchdims,nsamples, num_actions, 1])

  # -----------------------
  # policy conditioning
  # -----------------------
  # gaussian (mean=task, var=.1I)
  pshape = task_sampling.shape # [?, B, N, D]
  policies =  task_sampling + jnp.sqrt(var) * jax.random.normal(key, pshape)
  policies = policies.astype(dtype)
  policies = hk.BatchApply(policyfn, num_dims=num_dims)(policies)

  # input for SF
  sf_input = jnp.concatenate((state, policies), axis=-1)

  # -----------------------
  # compute successor features
  # -----------------------

  sf = hk.BatchApply(successorfn, num_dims=num_dims)(sf_input)
  sf = jnp.reshape(sf, [*sf.shape[:-1], num_actions, state_dim])

  # -----------------------
  # compute Q values
  # -----------------------
  q_values = jnp.sum(sf*policy_zeds, axis=-1)

  # -----------------------
  # GPI, best policy
  # -----------------------
  q_values = jnp.max(q_values, axis=policy_axis)

  return dict(
    sf=sf,
    policy_zeds=policy_zeds,
    q=q_values)

# ======================================================
# Networks
# ======================================================
class VisionTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, flatten=True):
    super().__init__(name='atari_torso')
    self._network = hk.Sequential([
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
        hk.Conv2D(16, [1, 1], 1)
    ])

    self.flatten = flatten

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    return jnp.reshape(outputs, [-1])  # [D]

class R2D2Network(hk.RNNCore):
  """A duelling recurrent network.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
      num_actions: int,
      lstm_size : int = 256,
      hidden_size : int=128,):
    super().__init__(name='r2d2_network')
    self._embed = embedding.OAREmbedding(
      VisionTorso(),
      num_actions)
    self._core = hk.LSTM(lstm_size)
    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[hidden_size])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[Predictions, hk.LSTMState]:
    del key
    image, task, state_feat, _, _ = process_inputs(inputs)
    embeddings = self._embed(inputs._replace(observation=image))  # [B, D+A+1]
    embeddings = jnp.concatenate([embeddings, state_feat], axis=-1)

    core_outputs, new_state = self._core(embeddings, state)
    # "UVFA"
    core_outputs = jnp.concatenate((core_outputs, task), axis=-1)
    q_values = self._duelling_head(core_outputs)
    return Predictions(q=q_values), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[Predictions, hk.LSTMState]:
    del key
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image, task, state_feat, _, _ = process_inputs(inputs)
    embeddings = hk.BatchApply(self._embed)(inputs._replace(observation=image))  # [T, B, D+A+1]
    embeddings = jnp.concatenate([embeddings, state_feat], axis=-1)
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    # "UVFA"
    core_outputs = jnp.concatenate((core_outputs, task), axis=-1)
    q_values = hk.BatchApply(self._duelling_head)(core_outputs)  # [T, B, A]
    return Predictions(q=q_values), new_states

class USFANetwork(hk.RNNCore):
  """Universal Successor Feature Approximators

  See https://arxiv.org/abs/1812.07626 for more information.
  """

  def __init__(self,
        num_actions: int,
        state_dim: int,
        lstm_size : int = 256,
        hidden_size : int=128,
        policy_size : int=32,
        variance: float=0.1,
        nsamples: int=30,
    ):
    super().__init__(name='usfa_network')
    self.var = variance
    self.nsamples = nsamples
    self.state_dim = state_dim
    self.num_actions = num_actions
    self.policy_size = policy_size
    self.hidden_size = hidden_size

    self.conv = VisionTorso()
    self.memory = hk.LSTM(lstm_size)
    self.statefn = hk.nets.MLP(
        [hidden_size],
        activate_final=True)

    self.policyfn = hk.nets.MLP(
        [policy_size, policy_size])

    self.successorfn = hk.nets.MLP([
        self.policy_size+self.hidden_size,
        self.num_actions*self.state_dim
        ])


  def usfa(self, memory_out, mem_state, task, state_feat, key, nbatchdims):
    """
    1. Sample K policy embeddings according to state features
    2. Compute corresponding successor features
    3. compute policy and do GPI

    Args:
        memory_out (TYPE): e.g. LSTM hidden
        mem_state (TYPE): e.g. full LSTM state
        task (TYPE): task vectors
        state_feat (TYPE): state features
        nbatchdims (TYPE): number of dimensions before data. E.g. [T,B]=2
    
    Returns:
        TYPE: Description
    """
    dtype = memory_out.dtype
    batchdims = [1]*nbatchdims
    policy_axis = nbatchdims

    if nbatchdims == 1:
      batch_apply = lambda x:x
    else:
      batch_apply = hk.BatchApply
    num_dims = nbatchdims + 1 # for policy axis


    state = batch_apply(self.statefn)(memory_out)
    # -----------------------
    # add dim for K=nsamples of policy params
    # tile along that dimension
    # -----------------------
    state = jnp.expand_dims(state, axis=policy_axis)
    task_sampling = jnp.expand_dims(task, axis=policy_axis)
    # add one extra for action
    policy_zeds = jnp.expand_dims(task, axis=(policy_axis,policy_axis+1))
    state = jnp.tile(state, [*batchdims,self.nsamples,1])
    task_sampling = jnp.tile(task_sampling, [*batchdims,self.nsamples,1])
    policy_zeds = jnp.tile(policy_zeds, [*batchdims,self.nsamples, self.num_actions, 1])

    # -----------------------
    # policy conditioning
    # -----------------------
    # gaussian (mean=task, var=.1I)
    # key = hk.next_rng_key()
    pshape = task_sampling.shape # [?, B, N, D]
    policies =  task_sampling + jnp.sqrt(self.var) * jax.random.normal(key, pshape)
    policies = policies.astype(dtype)
    policies = hk.BatchApply(self.policyfn, num_dims=num_dims)(policies)

    # input for SF
    sf_input = jnp.concatenate((state, policies), axis=-1)

    # -----------------------
    # compute successor features
    # -----------------------

    sf = hk.BatchApply(self.successorfn, num_dims=num_dims)(sf_input)
    sf = jnp.reshape(sf, [*sf.shape[:-1], self.num_actions, self.state_dim])

    # -----------------------
    # compute Q values
    # -----------------------
    q_values = jnp.sum(sf*policy_zeds, axis=-1)

    # -----------------------
    # GPI, best policy
    # -----------------------
    q_values = jnp.max(q_values, axis=policy_axis)

    return USFAPreds(
      sf=sf,
      policy_zeds=policy_zeds,
      q=q_values)


  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory.initial_state(batch_size)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Predictions, hk.LSTMState]:

    image, task, state_feat, last_action, last_reward = process_inputs(inputs)
    # -----------------------
    # compute state
    # -----------------------
    conv = self.conv(image)
    mem_inputs = jnp.concatenate((conv, last_action, last_reward), axis=-1)
    memory_out, mem_state = self.memory(mem_inputs, state)

    preds = self.usfa(memory_out, mem_state, task, state_feat, key, nbatchdims=1)

    return preds, mem_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      mem_state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Predictions, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image, task, state_feat, last_action, last_reward = process_inputs(inputs)

    # -----------------------
    # compute state
    # -----------------------
    conv = hk.BatchApply(self.conv)(image)  # [T, B, D+A+1]
    mem_inputs = jnp.concatenate((conv, last_action, last_reward), axis=2)
    memory_out, new_mem_state = hk.static_unroll(
      self.memory,
      mem_inputs,
      mem_state)

    preds = self.usfa(memory_out, new_mem_state, task, state_feat, key, nbatchdims=2)

    return preds, new_mem_state


class USFARewardNetwork(USFANetwork):
  """Universal Successor Feature Approximators

  See https://arxiv.org/abs/1812.07626 for more information.
  """
  def usfa(self, memory_out, mem_state, task, state_feat, key, nbatchdims):

    preds = super().usfa(memory_out, mem_state, task, state_feat, key, nbatchdims)

    if nbatchdims == 1:
      batch_apply = lambda x:x
    else:
      batch_apply = hk.BatchApply
    num_dims = nbatchdims + 1 # for policy axis

    cumfn = hk.nets.MLP(
        [self.hidden_size, self.state_dim],
        activate_final=False)

    cumulants = cumfn(memory_out)

    return USFAUnsupPreds(**preds._asdict(), cumulants=cumulants)




class UsfaFarmMixture(hk.RNNCore):
  """Universal Successor Feature Approximators + Feature Attending Recurrent Modules

  See https://arxiv.org/abs/1812.07626 for more information.
  """

  def __init__(self,
        num_actions: int,
        state_dim: int,
        lstm_size : int = 128,
        hidden_size : int=128,
        policy_size : int=32,
        variance: float=0.1,
        nsamples: int=30,
        nmodules: int=4,
    ):
    super().__init__(name='usfa_farm_network')
    self.var = variance
    self.nsamples = nsamples
    self.state_dim = state_dim
    self.num_actions = num_actions
    self.policy_size = policy_size
    self.hidden_size = hidden_size

    self.conv = VisionTorso(flatten=False)
    self.memory = FARM(lstm_size, nmodules)
    self.statefn = hk.nets.MLP(
        [hidden_size],
        activate_final=True)

    self.policynet = hk.nets.MLP(
        [policy_size, policy_size])

    self.successorfn = hk.nets.MLP([
        self.policy_size+self.hidden_size,
        self.num_actions*self.state_dim
        ])

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory.initial_state(batch_size)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Predictions, hk.LSTMState]:

    image, task, state_feat, last_action, last_reward = process_inputs(inputs)
    # -----------------------
    # compute state
    # -----------------------
    conv = self.conv(image)
    mem_inputs = jnp.concatenate((conv, last_action, last_reward), axis=-1)
    memory_out, mem_state = self.memory(mem_inputs, state)

    preds = self.usfa(memory_out, mem_state, task, state_feat, key, nbatchdims=1)

    return preds, mem_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      mem_state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[Predictions, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image, task, state_feat, last_action, last_reward = process_inputs(inputs)

    # -----------------------
    # compute state
    # -----------------------
    conv = hk.BatchApply(self.conv)(image)  # [T, B, H, W, C]
    mem_inputs = FarmInputs(
      image=conv,
      vector=jnp.concatenate((last_action, last_reward), axis=2)
      )

    # [T, B, N_H, D]
    memory_out, new_mem_state = hk.static_unroll(
      self.memory,
      mem_inputs,
      mem_state)

    # -----------------------
    # compute policy embeddings
    # -----------------------
    nbatchdims = 2
    policy_zs = sample_gauss(
      mean=task, var=self.var, nsamples=self.nsamples,
      key=key, samples_axis=nbatchdims)
    # num_dims=policy_axis+1 indicates [?, B, N_P]
    policies = hk.BatchApply(self.policynet, num_dims=nbatchdims+1)(policy_zs)

    # -----------------------
    # compute SF inputs
    # -----------------------
    # mixture
    mem_weights = hk.Linear(self.memory.nmodules, with_bias=False)(policies) # [T, B, N, N_H]

    import ipdb; ipdb.set_trace()
    # jnp.dot(mem_weights, memory_out.transpose(0,1,3,2))
    # mem_sf = mem_weights



    preds = usfa(memory_out,
        task=task,
        statefn=self.statefn,
        policyfn=self.policynet,
        successorfn=self.successorfn,
        num_actions=self.num_actions,
        nsamples=self.nsamples,
        state_dim=self.state_dim,
        var=self.var,
        key=key,
        nbatchdims=nbatchdims)



    return preds, new_mem_state
