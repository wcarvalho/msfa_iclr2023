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


# ======================================================
# TYpes
# ======================================================
Images = jnp.ndarray

class USFAPredictions(NamedTuple):
  q: jnp.ndarray
  sf: jnp.ndarray
  policy_zeds: jnp.ndarray


class USFARewardPredictions(NamedTuple):
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
# Networks
# ======================================================
class VisionTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self):
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

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)

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
    return q_values, new_state

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
    return q_values, new_states

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


  def usfa(self, mem_outputs, mem_state, task, state_feat, key, nbatchdims):
    """
    1. Sample K policy embeddings according to state features
    2. Compute corresponding successor features
    3. compute policy and do GPI

    Args:
        mem_outputs (TYPE): e.g. LSTM hidden
        mem_state (TYPE): e.g. full LSTM state
        task (TYPE): task vectors
        state_feat (TYPE): state features
        nbatchdims (TYPE): number of dimensions before data. E.g. [T,B]=2
    
    Returns:
        TYPE: Description
    """
    dtype = mem_outputs.dtype
    batchdims = [1]*nbatchdims
    policy_axis = nbatchdims

    if nbatchdims == 1:
      batch_apply = lambda x:x
    else:
      batch_apply = hk.BatchApply
    num_dims = nbatchdims + 1 # for policy axis


    state = batch_apply(self.statefn)(mem_outputs)
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

    return USFAPredictions(
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
    mem_outputs, mem_state = self.memory(mem_inputs, state)

    preds = self.usfa(mem_outputs, mem_state, task, state_feat, key, nbatchdims=1)

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
    mem_outputs, new_mem_state = hk.static_unroll(
      self.memory,
      mem_inputs,
      mem_state)

    preds = self.usfa(mem_outputs, new_mem_state, task, state_feat, key, nbatchdims=2)

    return preds, new_mem_state



class USFARewardNetwork(USFANetwork):
  """Universal Successor Feature Approximators

  See https://arxiv.org/abs/1812.07626 for more information.
  """
  def usfa(self, mem_outputs, mem_state, task, state_feat, key, nbatchdims):

    preds = super().usfa(mem_outputs, mem_state, task, state_feat, key, nbatchdims)

    if nbatchdims == 1:
      batch_apply = lambda x:x
    else:
      batch_apply = hk.BatchApply
    num_dims = nbatchdims + 1 # for policy axis

    cumfn = hk.nets.MLP(
        [self.hidden_size, self.state_dim],
        activate_final=True,
        activation=jax.nn.sigmoid)

    cumulants = cumfn(mem_outputs)

    return USFARewardPredictions(**preds._asdict(), cumulants=cumulants)
