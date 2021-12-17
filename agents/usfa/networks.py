from typing import Tuple

from acme import types
from acme.jax import networks as networks_lib
from acme.jax.networks import base
from acme.jax.networks import embedding
from acme.wrappers import observation_action_reward


import haiku as hk
import jax
import jax.numpy as jnp

def process_inputs(inputs):
  last_action = inputs.action
  last_reward = inputs.reward
  observation = inputs.observation

  image = observation.image
  task = observation.task
  state_feat = observation.state_features


  last_action = jnp.expand_dims(last_action, -1)
  last_reward = jnp.expand_dims(last_reward, -1)

  return image, task, state_feat, last_action, last_reward

class USFANetwork(hk.RNNCore):
  """Universal Successor Feature Approximators

  See https://arxiv.org/abs/1812.07626 for more information.
  """

  def __init__(self,
        num_actions: int,
        lstm_size : int=256,
        hidden_size : int=128,
        policy_size : int=32,
        variance: float=0.1,
        nsamples: int=30,
    ):
    super().__init__(name='usfa_network')
    self.var = variance
    self.nsamples = nsamples
    self.num_actions = num_actions
    self.policy_size = policy_size
    self.hidden_size = hidden_size

    self.conv = networks_lib.AtariTorso()
    self.memory = hk.LSTM(lstm_size)
    self.statefn = hk.nets.MLP(
        [hidden_size],
        activate_final=True)

    self.policyfn = hk.nets.MLP(
        [policy_size, policy_size])



  def usfa(self, state, task, state_feat, key, nbatchdims):
    batchdims = [1]*nbatchdims
    policy_axis = nbatchdims

    # -----------------------
    # add dim for K=nsamples of policy params
    # tile along that dimension
    # -----------------------
    state = jnp.expand_dims(state, axis=policy_axis)
    task_sampling = jnp.expand_dims(task, axis=policy_axis)
    # add one extra for action
    task_action = jnp.expand_dims(task, axis=(policy_axis,policy_axis+1))
    state = jnp.tile(state, [*batchdims,self.nsamples,1])
    task_sampling = jnp.tile(task_sampling, [*batchdims,self.nsamples,1])
    task_action = jnp.tile(task_action, [*batchdims,self.nsamples, self.num_actions, 1])

    # -----------------------
    # policy conditioning
    # -----------------------
    # gaussian (mean=task, var=.1I)
    policies =  task_sampling + jnp.sqrt(self.var) * jax.random.normal(key, task_sampling.shape)
    policies = hk.BatchApply(self.policyfn)(policies)

    # input for SF
    sf_input = jnp.concatenate((state, policies), axis=-1)

    # -----------------------
    # compute successor features
    # -----------------------
    self.successorfn = hk.nets.MLP([
        self.policy_size+self.hidden_size,
        self.num_actions*state_feat.shape[-1]
        ])

    sf = hk.BatchApply(self.successorfn)(sf_input)
    sf = jnp.reshape(sf, [*sf.shape[:-1], self.num_actions, state_feat.shape[-1]])

    # -----------------------
    # compute Q values
    # -----------------------
    q_values = jnp.sum(sf*task_action, axis=-1)

    # -----------------------
    # GPI, best policy
    # -----------------------
    return jnp.max(q_values, axis=policy_axis)


  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[base.QValues, hk.LSTMState]:

    image, task, state_feat, last_action, last_reward = process_inputs(inputs)
    # -----------------------
    # compute state
    # -----------------------
    conv = self.conv(image)
    mem_inputs = jnp.concatenate((conv, last_action, last_reward), axis=-1)
    mem_outputs, mem_state = self.memory(mem_inputs, state)
    state = self.statefn(mem_outputs)


    start = [1]
    q_values = self.usfa(state, task, state_feat, key, nbatchdims=1)

    return q_values, mem_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      mem_state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    image, task, state_feat, last_action, last_reward = process_inputs(inputs)

    # -----------------------
    # compute state
    # -----------------------
    conv = hk.BatchApply(self.conv)(image)  # [T, B, D+A+1]
    mem_inputs = jnp.concatenate((conv, last_action, last_reward), axis=2)
    mem_outputs, new_mem_state = hk.static_unroll(self.memory, mem_inputs, mem_state)
    state = hk.BatchApply(self.statefn)(mem_outputs)

    q_values = self.usfa(state, task, state_feat, key, nbatchdims=2)

    return q_values, new_mem_state

    # # -----------------------
    # # policy conditioning
    # # -----------------------
    # # gaussian (mean=task, var=.1I)
    # task_samples = jnp.tile(jnp.expand_dims(task, axis=2), [1,1,self.nsamples,1])
    # policies =  task_samples + jnp.sqrt(self.var) * jax.random.normal(key, task_samples.shape)
    # policies = hk.BatchApply(self.policyfn)(policies)

    # # input for SF
    # state = jnp.tile(jnp.expand_dims(state, axis=2), [1,1, self.nsamples, 1])
    # sf_input = jnp.concatenate((state, policies), axis=3)

    # # -----------------------
    # # compute successor features
    # # -----------------------
    # self.successorfn = hk.nets.MLP([
    #     self.policy_size+self.hidden_size,
    #     self.num_actions*state_feat.shape[-1]
    #     ])

    # sf = hk.BatchApply(self.successorfn)(sf_input)
    # sf = jnp.reshape(sf, [*sf.shape[:-1], self.num_actions, state_feat.shape[-1]])

    # # -----------------------
    # # compute Q values
    # # -----------------------
    # task_expand = jnp.tile(jnp.expand_dims(task, axis=(2,3)), [1,1,self.nsamples, self.num_actions, 1])
    # q_values = jnp.sum(sf*task_expand, axis=-1)

    # # -----------------------
    # # GPI
    # # -----------------------
    # # best policy
    # q_values = jnp.max(q_values, axis=2)