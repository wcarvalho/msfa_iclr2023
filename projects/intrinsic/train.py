"""Simple intrinsic reward on BabyAI derivative environment.

CUDA_VISIBLE_DEVICES=0 \
  BABYAI_STORAGE='data/' \
  python projects/intrinsic/train.py
"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from absl import app
from absl import flags
import acme
import functools
import functools
from functools import partial
from typing import Callable, Iterator, List, Optional
from typing import Tuple

import acme
import bsuite
import dataclasses
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax
import tree
from absl import app
from absl import flags
from acme import adders
from acme import core
from acme import environment_loop
from acme import specs
from acme import types
from acme import wrappers
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax import r2d2
from acme.agents.jax.dqn import learning_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax import utils
from acme.jax import utils as jax_utils
from acme.jax import variable_utils
from acme.jax.layouts import local_layout
from acme.jax.networks import base
from acme.jax.networks import embedding
from acme.jax.networks import duelling
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import observation_action_reward


from agents import td_agent
from projects.offline_sf import helpers
from projects.msf import networks as msf_networks
from utils import make_logger, gen_log_dir

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')

FLAGS = flags.FLAGS

def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)


# ======================================================
# Learning objective
# ======================================================

@dataclasses.dataclass
class LossFn_with_RND(learning_lib.LossFn):
  """R2D2 Learning."""
  discount: float = 0.99
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # More than DQN
  max_replay_size: int = 1_000_000
  store_lstm_state: bool = True
  max_priority_weight: float = 0.9
  bootstrap_n: int = 5
  importance_sampling_exponent: float = 0.2

  burn_in_length: int = None
  clip_rewards : bool = False
  max_abs_reward: float = 1.

  def td_error(self, data, online_q, online_state, target_q, target_state):
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_q, axis=-1)
    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype)
    rewards = data.reward
    if self.clip_rewards:
      rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)
    rewards = rewards.astype(online_q.dtype)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n,
            tx_pair=self.tx_pair),
        in_axes=1,
        out_axes=1)
    batch_td_error = batch_td_error_fn(
        online_q[:-1],
        data.action[:-1],
        target_q[1:],
        selector_actions[1:],
        rewards[:-1],
        discounts[:-1])
    batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)
    return batch_td_error, batch_loss


  def rnd_error(self, y1, y2):
    y2 = jax.lax.stop_gradient(y2)
    error = y1 - y2

    return error.mean(-1)

  def __call__(
      self,
      network,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key_grad: networks_lib.PRNGKey,
    ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""

    unroll = network.unroll  # convenienve

    # Convert sample data to sequence-major format [T, B, ...].
    data = jax_utils.batch_to_sequence(batch.data)

    # Get core state & warm it up on observations for a burn-in period.
    if self.store_lstm_state:
      # Replay core state.
      online_state = jax.tree_map(lambda x: x[0], data.extras['core_state'])
    else:
      _, batch_size = data.action.shape
      key_grad, key = jax.random.split(key_grad)
      online_state = network.initial_state.apply(params, key, batch_size)
    target_state = online_state

    # Maybe burn the core state in.
    burn_in_length = self.burn_in_length
    if burn_in_length:
      burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
      key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
      _, _, _, online_state = unroll.apply(params, key1, burn_obs, online_state)
      key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
      _, _, _, target_state = unroll.apply(target_params, key1, burn_obs,
                                     target_state)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    # Unroll on sequences to get online and target Q-Values.

    key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
    online_q, online_y1, online_y2, online_state = unroll.apply(params, key1, data.observation, online_state)
    key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
    target_q, _, _, target_state = unroll.apply(target_params, key1, data.observation,
                               target_state)

    rnd_error = self.rnd_error(online_y1, online_y2)

    # add intrinsic reward
    data = data._replace(reward=data.reward + jax.lax.stop_gradient(rnd_error))

    batch_td_error, batch_loss = self.td_error(data, online_q, online_state, target_q, target_state)



    # Importance weighting.
    probs = batch.info.probability
    importance_weights = (1. / (probs + 1e-6)).astype(online_q.dtype)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    mean_loss = jnp.mean(importance_weights * batch_loss)

    # loss adter intrinsic
    mean_loss = mean_loss + 1e-3*rnd_error.mean()

    # Calculate priorities as a mixture of max and mean sequence errors.
    abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = (max_priority + mean_priority)

    reverb_update = learning_lib.ReverbUpdate(
        keys=batch.info.key,
        priorities=priorities
        )
    extra = learning_lib.LossExtra(metrics={'rnd': rnd_error.mean()}, reverb_update=reverb_update)
    return mean_loss, extra

# ======================================================
# Network
# ======================================================
class RND_QNetwork(hk.RNNCore):
  """A duelling recurrent network + RND
  """

  def __init__(self,
      num_actions: int,
      lstm_size : int = 256,
      hidden_size : int=128,):
    super().__init__(name='r2d2_network')
    self._embed = embedding.OAREmbedding(
      msf_networks.VisionTorso(),
      num_actions)
    self._core = hk.LSTM(lstm_size)
    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[hidden_size])

    self.predictor1 = hk.nets.MLP([100, 100, 100])
    self.predictor2 = hk.nets.MLP([100, 100, 100])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      key: networks_lib.PRNGKey,
    ) -> Tuple[base.QValues, hk.LSTMState]:
    del key

    embeddings = self._embed(inputs._replace(observation=inputs.observation.image))  # [B, D+A+1]

    # -----------------------
    # RND computations
    # -----------------------
    y1 = self.predictor1(embeddings)
    y2 = self.predictor2(embeddings)


    # -----------------------
    # R2D2 computations
    # -----------------------
    core_outputs, new_state = self._core(embeddings, state)
    # "UVFA"
    q_values = self._duelling_head(core_outputs)
    return q_values, y1, y2, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
      key: networks_lib.PRNGKey,
  ) -> Tuple[base.QValues, hk.LSTMState]:
    del key
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    embeddings = hk.BatchApply(self._embed)(inputs._replace(observation=inputs.observation.image))  # [T, B, D+A+1]

    # -----------------------
    # RND computations
    # -----------------------
    y1 = hk.BatchApply(self.predictor1)(embeddings)
    y2 = hk.BatchApply(self.predictor2)(embeddings)

    # -----------------------
    # R2D2 computations
    # -----------------------
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    # "UVFA"
    q_values = hk.BatchApply(self._duelling_head)(core_outputs)  # [T, B, A]
    return q_values, y1, y2, new_states

# ======================================================
# Custom Behavior policy constructor.
# ======================================================
def make_behavior_policy(
    networks: td_agent.TDNetworkFns,
    config: td_agent.R2D1Config,
    evaluation: bool = False,
    ) -> r2d2_networks.EpsilonRecurrentPolicy:
  """Selects action according to the policy.
  
  Args:
      networks (td_agent.TDNetworkFns): Network functions
      config (R2D1Config): Config
      evaluation (bool, optional): whether evaluation policy
      network_samples (bool, optional): whether network is random
  
  Returns:
      r2d2_networks_lib.EpsilonRecurrentPolicy: epsilon-greedy policy
  """

  def behavior_policy(
                      params: networks_lib.Params,
                      key: networks_lib.PRNGKey,
                      observation,
                      core_state,
                      epsilon):
    key, key_net, key_sample = jax.random.split(key, 3)
    q_values, y1, y2, core_state = networks.forward.apply(
        params, key_net, observation, core_state, key_sample)
    epsilon = config.evaluation_epsilon if evaluation else epsilon
    return rlax.epsilon_greedy(epsilon).sample(key_net, q_values), core_state

  return behavior_policy



def main(_):
  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/intrinsic/local",
    agent=FLAGS.agent)
  logger_fn = lambda : make_logger(
        log_dir=log_dir, label=f'{FLAGS.agent}')


  # -----------------------
  # environment
  # -----------------------
  env = helpers.make_environment(
    task_kinds=['place'],
    partial_obs=True)
  env_spec = acme.make_environment_spec(env)

  # -----------------------
  # agent
  # -----------------------
  default_config = dict(
    # network
    discount=0.99,

    # Learner options
    trace_length=40,
    learning_rate=5e-5,
    max_number_of_steps=5_000_000, # 5M takes 1hr

    # How many gradient updates to perform per learner step.
    num_sgd_steps_per_step=4,

    # Replay options
    batch_size=32,
    min_replay_size=10_000,
    max_replay_size=100_000,
    )
  config = td_agent.R2D1Config(**default_config)

  NetworkCls=RND_QNetwork
  NetKwargs=dict(
    num_actions=env_spec.actions.num_values,
    )
  LossFn = LossFn_with_RND


  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn,
      logger_fn=logger_fn)
  agent = td_agent.TDAgent(
      env_spec,
      behavior_policy_constructor=make_behavior_policy,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs),
      builder=builder,
      workdir=log_dir,
      config=config,
      seed=FLAGS.seed,
      )

  # -----------------------
  # make env + run
  # -----------------------
  env_logger = make_logger(
    log_dir=log_dir,
    label='actor',
    steps_key="steps")

  loop = acme.EnvironmentLoop(env, agent, logger=env_logger)
  loop.run(FLAGS.num_episodes)



if __name__ == '__main__':
  app.run(main)
