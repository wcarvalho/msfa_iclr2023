"""
PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python -m ipdb -c continue projects/leaks/test_kitch_arch_tuple.py

PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess projects/leaks/test_kitch_arch_tuple.py

"""


import functools
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple

import collections
import acme
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

# from absl import app
# from acme import adders
from acme import core
# from acme import environment_loop
from acme import specs
# from acme import types
from acme import wrappers
from acme.agents.jax import r2d2
from acme.agents.jax.dqn import learning_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import networks as r2d2_networks
# from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
# from acme.jax import types
from acme.jax import utils
# from acme.jax import utils as jax_utils
# from acme.jax import variable_utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.jax.networks import base
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import GymWrapper

from babyai.levels.iclr19_levels import Level_GoToRedBallGrey
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

NUM_EPISODES = int(100e6)
SEED = 0
DISTRIBUTED = True

from projects.leaks.mini_lib import make_networks, TDAgent, DistributedTDAgent, TDBuilder, R2D2Learning

from projects.msf import helpers
from agents import td_agent

@dataclasses.dataclass
class R2D2LearningTuple(R2D2Learning):
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

  def error(self, data, online_q, online_state, target_q, target_state):
    # Get value-selector actions from online Q-values for double Q-learning.
    online_q = online_q.q
    target_q = target_q.q

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
    data = utils.batch_to_sequence(batch.data)

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
      _, online_state = unroll.apply(params, key1, burn_obs, online_state)
      key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
      _, target_state = unroll.apply(target_params, key1, burn_obs,
                                     target_state)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    # Unroll on sequences to get online and target Q-Values.

    key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
    online_q, online_state = unroll.apply(params, key1, data.observation, online_state)
    key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
    target_q, target_state = unroll.apply(target_params, key1, data.observation,
                               target_state)

    batch_td_error, batch_loss = self.error(data, online_q, online_state, target_q, target_state)

    # Importance weighting.
    probs = batch.info.probability
    importance_weights = (1. / (probs + 1e-6)).astype(online_q.q.dtype)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    mean_loss = jnp.mean(importance_weights * batch_loss)

    # Calculate priorities as a mixture of max and mean sequence errors.
    abs_td_error = jnp.abs(batch_td_error).astype(online_q.q.dtype)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = (max_priority + mean_priority)

    reverb_update = learning_lib.ReverbUpdate(
        keys=batch.info.key,
        priorities=priorities
        )
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return mean_loss, extra


def make_behavior_policy(
    networks,
    config,
    evaluation: bool = False,
    ):
  """Selects action according to the policy.
  
  Args:
      networks (TDNetworkFns): Network functions
      config (R2D1Config): Config
      evaluation (bool, optional): whether evaluation policy
      network_samples (bool, optional): whether network is random
  
  Returns:
      r2d2_networks_lib.EpsilonRecurrentPolicy: epsilon-greedy policy
  """

  def behavior_policy(
                      params,
                      key,
                      observation,
                      core_state,
                      epsilon):
    key, key_net, key_sample = jax.random.split(key, 3)

    # -----------------------
    # if evaluating & have seperation evaluation function, use it
    # -----------------------
    forward_fn = networks.forward.apply
    preds, core_state = forward_fn(
        params, key_net, observation, core_state)
    epsilon = config.evaluation_epsilon if evaluation else epsilon
    return rlax.epsilon_greedy(epsilon).sample(key_net, preds.q),core_state

  return behavior_policy
# ======================================================
# main functions
# ======================================================
def train():
  env = helpers.make_environment()
  env_spec = specs.make_environment_spec(env)

  config, NetworkCls, NetKwargs, _, _, _, _ = helpers.load_agent_settings(agent='r2d1', env_spec=env_spec, config_kwargs=None)

  config.batch_size = 32
  config.samples_per_insert_tolerance_rate = 0.1
  config.samples_per_insert = 0.0 # different
  config.min_replay_size = 1_000 # smaller
  config.max_replay_size = 10_000 # smaller
  config.max_gradient_norm = 80
  config.num_sgd_steps_per_step = 4
  config.max_number_of_steps = 100_000_000


  BuilderCls= functools.partial(TDBuilder,
    loss_fn=R2D2LearningTuple,
    )
  agent = TDAgent(
      env_spec,
      BuilderCls=BuilderCls,
      networks=make_networks(config.batch_size, env_spec, 
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs),
      config=config,
      behavior_constructor=make_behavior_policy,
      workdir='./results/babyai_kitchen/',
      seed=SEED)

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(NUM_EPISODES)

def distributed():
  environment_factory = lambda key: helpers.make_environment()
  env = environment_factory(0)
  env_spec = acme.make_environment_spec(env)
  del env

  config, NetworkCls, NetKwargs, _, _, _, _ = helpers.load_agent_settings(agent='r2d1', env_spec=env_spec, config_kwargs=None)

  config.batch_size = 32
  config.burn_in_length = 0
  config.trace_length = 20 # shorter
  config.sequence_period = 40
  config.prefetch_size = 0
  config.samples_per_insert_tolerance_rate = 0.1
  config.samples_per_insert = 0.0 # different
  config.num_parallel_calls = 1
  config.min_replay_size = 1_000 # smaller
  config.max_replay_size = 10_000 # smaller
  config.max_gradient_norm = 80
  config.num_sgd_steps_per_step = 4

  num_actions = env_spec.actions.num_values
  def network_factory(spec):
    return make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs)


  BuilderCls= functools.partial(TDBuilder,
    loss_fn=R2D2LearningTuple,
    )
  program = DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      behavior_constructor=make_behavior_policy,
      network_factory=network_factory,
      config=config,
      BuilderCls=BuilderCls,
      seed=1,
      num_actors=1,
      max_number_of_steps=10e6,
      log_every=30.0).build()

  import launchpad as lp
  from launchpad.nodes.python.local_multi_processing import PythonProcess

  controller = lp.launch(program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
  )
  controller.wait()

def main():
  if DISTRIBUTED:
    distributed()
  else:
    train()


if __name__ == '__main__':
  main()
