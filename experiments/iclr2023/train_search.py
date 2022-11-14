"""
Param search.
"""

import functools
from absl import app
from absl import flags
from pathlib import Path
from hyperopt import hp
import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess
# from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import ParameterGrid
from ray import tune
import multiprocessing as mp
import jax
import time
from pprint import pprint
from utils import gen_log_dir
import os
import importlib

from experiments.common.train_search import run_experiments, listify_space
from experiments.iclr2023.train_distributed import build_program

FLAGS = flags.FLAGS

def main(_):
  """This will select `FLAGS.search` from the `FLAGS.spaces` file and run it on the designated GPUs.
  
  Args:
      _ (TYPE): Description
  """
  FLAGS = flags.FLAGS
  terminal = FLAGS.terminal
  mp.set_start_method('spawn')

  assert FLAGS.search != '', 'set search!'
  space = importlib.import_module(f'projects.iclr2023.{FLAGS.spaces}').get(FLAGS.search, FLAGS.agent)

  if FLAGS.idx is not None:
    listify_space(space)
    if FLAGS.idx < len(space):
      space = space[FLAGS.idx]
    else:
      return

  # root_path is needed to tell program absolute path
  # this is used for BabyAI
  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder if FLAGS.folder else f"results/{FLAGS.env}"
  use_date = FLAGS.date
  use_wandb = FLAGS.wandb
  group = FLAGS.group if FLAGS.group else FLAGS.search # overall group
  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=group, # overall group
    notes=FLAGS.notes,
    save_code=True,
  )

  if FLAGS.env == "goto":
    default_env_kwargs=dict(setting='xl_respawn')
  elif FLAGS.env == "fruitbot":
    default_env_kwargs=dict(
      setting='easy',
      max_episodes=4,
      completion_bonus=0.0,
      env_reward_coeff=1.0,
      env_task_dim=2)
  elif FLAGS.env == "minihack":
    default_env_kwargs=dict(
      setting='room',
      num_train_seeds=200)
  else:
    raise NotImplementedError(FLAGS.env)



  run_experiments(
    build_program_fn=functools.partial(build_program,
      env=FLAGS.env),
    space=space,
    root_path=root_path,
    folder=folder,
    group=group,
    wandb_init_kwargs=wandb_init_kwargs,
    default_env_kwargs=default_env_kwargs,
    use_wandb=use_wandb,
    terminal=FLAGS.terminal,
    skip=FLAGS.skip,
    use_ray=FLAGS.ray,
    debug=FLAGS.debug_search)


if __name__ == '__main__':
  app.run(main)
