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

from absl import app
from absl import flags

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

from projects.leaks import mini_lib_tuple as mlt_agent
from projects.msf import helpers
from agents import td_agent



flags.DEFINE_bool('evaluator', True, '')
flags.DEFINE_string('env', 'babyai', '')
flags.DEFINE_string('netfn', 'minilib', '')
flags.DEFINE_string('behavior_policy', 'minilib', '')
flags.DEFINE_string('loss_fn', 'minilib', '')
flags.DEFINE_string('make_nets', 'minilib', '')
flags.DEFINE_string('builder', 'default', '')
flags.DEFINE_string('program', 'default', '')
flags.DEFINE_integer('clear_cache', 0, '')
flags.DEFINE_bool('cuda_actors', False, '')


FLAGS = flags.FLAGS

def distributed():
  # ======================================================
  # env
  # ======================================================
  if FLAGS.env == "babyai":
    make_env = mlt_agent.make_babyai_environment
  elif FLAGS.env == "bsuite":
    make_env = mlt_agent.make_babyai_environment
  elif FLAGS.env == "default":
    make_env = helpers.make_environment
  else:
    raise NotImplementedError

  environment_factory = lambda key: make_env()
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

  # ======================================================
  # Loss fn
  # ======================================================
  if FLAGS.loss_fn == "minilib":
    LossFn = mlt_agent.R2D2Learning
    LossFnKwargs=dict()
  elif FLAGS.loss_fn == "default":
    pass
  else:
    raise NotImplementedError

  # ======================================================
  # HK Network
  # ======================================================
  # if FLAGS.netfn == "minilib":
  #   NetworkCls = mlt_agent.r2d1
  if FLAGS.netfn == "minilib":
    from projects.leaks.mini_lib_tuple import SimpleRecurrentQNetwork
    NetworkCls = SimpleRecurrentQNetwork
    num_actions = env_spec.actions.num_values
    NetKwargs = dict(num_actions=num_actions, memory_size=512)
  elif FLAGS.netfn == "default":
    pass
  else:
    raise NotImplementedError

  # ======================================================
  # behavior policy
  # ======================================================
  if FLAGS.behavior_policy == "default":
    behavior_policy_constructor=td_agent.make_behavior_policy
  elif FLAGS.behavior_policy == "minilib":
    behavior_policy_constructor=mlt_agent.make_behavior_policy
  else:
    raise NotImplementedError


  # ======================================================
  # Network creater
  # ======================================================
  if FLAGS.make_nets == "default":
    make_nets = td_agent.make_networks
  elif FLAGS.make_nets == "minilib":
    make_nets = mlt_agent.make_networks
    behavior_policy_constructor=mlt_agent.make_behavior_policy
  else:
    raise NotImplementedError

  # ======================================================
  # Builder
  # ======================================================
  if FLAGS.builder == "default":
    BuilderCls = td_agent.TDBuilder
  elif FLAGS.builder == "minilib":
    BuilderCls = mlt_agent.TDBuilder
  else:
    raise NotImplementedError

  # ======================================================
  # Agent
  # ======================================================
  if FLAGS.program == "default":
    ProgramCls = td_agent.DistributedTDAgent
  elif FLAGS.program == "minilib":
    ProgramCls = mlt_agent.DistributedTDAgent
  else:
    raise NotImplementedError

  def print_flags(flags):
    from pprint import pprint
    print("="*50)
    pprint(flags)
    print("="*50)

  print_flags(dict(
    netfn=NetworkCls,
    behavior_policy=behavior_policy_constructor,
    env=make_env,
    loss_fn=LossFn,
    make_nets=make_nets,
    builder=BuilderCls,
    program=ProgramCls,
    ))
  # import ipdb; ipdb.set_trace()

  def network_factory(spec):
    return make_nets(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs)

  BuilderCls= functools.partial(BuilderCls,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    learner_kwargs=dict(
      clear_sgd_cache_period=FLAGS.clear_cache)
    )
  program = ProgramCls(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      behavior_policy_constructor=behavior_policy_constructor,
      network_factory=network_factory,
      config=config,
      builder=BuilderCls,
      seed=1,
      num_actors=1,
      evaluator=FLAGS.evaluator,
      max_number_of_steps=10e6,
      log_every=1.0).build()

  import launchpad as lp
  from launchpad.nodes.python.local_multi_processing import PythonProcess

  if FLAGS.cuda_actors:
    local_resources = None
  else:
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
  controller = lp.launch(program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    local_resources=local_resources,
  )
  controller.wait()

def main(_):
  """
  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/       LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/       CUDA_VISIBLE_DEVICES=1       XLA_PYTHON_CLIENT_PREALLOCATE=false       TF_FORCE_GPU_ALLOW_GROWTH=true       mprof run --multiprocess projects/leaks/test_kitch_distributed.py       --netfn minilib --behavior_policy minilib --env minilib --loss_fn minilib
  """
  if DISTRIBUTED:
    distributed()
  else:
    train()


if __name__ == '__main__':
  app.run(main)
