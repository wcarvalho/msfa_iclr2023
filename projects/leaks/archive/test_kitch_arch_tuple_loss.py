"""
PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python -m ipdb -c continue projects/leaks/test_kitch_arch_loss.py

PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess projects/leaks/test_kitch_arch_loss.py

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

from acme import core
from acme import specs
from acme import wrappers
from acme.agents.jax import r2d2
from acme.agents.jax.dqn import learning_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.jax.networks import base
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import GymWrapper


NUM_EPISODES = int(100e6)
SEED = 0
DISTRIBUTED = True

"""
rewritten:
  - 
  - make_behavior_policy
original:
  - 

"""
from projects.leaks.mini_lib import TDAgent, DistributedTDAgent, TDBuilder, R2D2Learning

from projects.msf import helpers
from agents import td_agent


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
# def train():
#   env = helpers.make_environment()
#   env_spec = specs.make_environment_spec(env)

#   config, NetworkCls, NetKwargs, _, _, _, _ = helpers.load_agent_settings(agent='r2d1', env_spec=env_spec, config_kwargs=None)

#   config.batch_size = 32
#   config.samples_per_insert_tolerance_rate = 0.1
#   config.samples_per_insert = 0.0 # different
#   config.min_replay_size = 1_000 # smaller
#   config.max_replay_size = 10_000 # smaller
#   config.max_gradient_norm = 80
#   config.num_sgd_steps_per_step = 4
#   config.max_number_of_steps = 100_000_000


#   BuilderCls= functools.partial(TDBuilder,
#     loss_fn=td_agent.R2D2Learning,
#     )
#   agent = TDAgent(
#       env_spec,
#       BuilderCls=BuilderCls,
#       networks=td_agent.make_networks(config.batch_size, env_spec, 
#         NetworkCls=NetworkCls,
#         NetKwargs=NetKwargs),
#       config=config,
#       behavior_constructor=make_behavior_policy,
#       workdir='./results/babyai_kitchen/',
#       seed=SEED)

#   loop = acme.EnvironmentLoop(env, agent)
#   loop.run(NUM_EPISODES)

def distributed():
  environment_factory = lambda key: helpers.make_environment()
  env = environment_factory(0)
  env_spec = acme.make_environment_spec(env)
  del env

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = helpers.load_agent_settings(agent='r2d1', env_spec=env_spec, config_kwargs=None)

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
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs)

  BuilderCls= functools.partial(TDBuilder,
    LossFn=LossFn, LossFnKwargs=LossFnKwargs,
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
