"""
PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run projects/leaks/test_babyai.py

PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess projects/leaks/test_babyai.py

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

from projects.leaks.mini_lib import make_environment, make_networks, SimpleRecurrentQNetwork, TDAgent, DistributedTDAgent

# ======================================================
# main functions
# ======================================================
def train():
  env = make_environment()
  env_spec = specs.make_environment_spec(env)


  config = r2d2.R2D2Config(
      batch_size=32,
      trace_length=20,
      burn_in_length=20,
      sequence_period=20,
      max_replay_size=10_000)


  agent = TDAgent(
      env_spec,
      networks=make_networks(config.batch_size, env_spec, 
        NetworkCls=SimpleRecurrentQNetwork,
        NetKwargs=dict(
          num_actions=env_spec.actions.num_values)),
      config=config,
      workdir='./results/babyai_regular/',
      seed=SEED)

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(NUM_EPISODES)

def distributed():
  environment_factory = lambda key: make_environment()
  env = environment_factory(0)
  env_spec = acme.make_environment_spec(env)
  del env

  config = r2d2.R2D2Config(
      batch_size=32,
      burn_in_length=0,
      trace_length=20,
      sequence_period=40,
      prefetch_size=0,
      samples_per_insert_tolerance_rate=0.1,
      samples_per_insert=0.0, # single process
      num_parallel_calls=1,
      min_replay_size=1_000,
      max_replay_size=10_000)
  config.max_gradient_norm = 80
  config.num_sgd_steps_per_step = 4

  num_actions = env_spec.actions.num_values
  def network_factory(spec):
    return make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=SimpleRecurrentQNetwork,
      NetKwargs=dict(num_actions=num_actions))

  # from agents import td_agent
  # builder=functools.partial(td_agent.TDBuilder,
  #     LossFn=R2D2Learning)
  # return td_agent.DistributedTDAgent(

  program = DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      config=config,
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
