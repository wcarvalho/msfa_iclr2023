"""
Param search.
"""


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

from projects.common.train_search import run_experiments, listify_space
from projects.kitchen_combo.train_distributed import build_program


flags.DEFINE_string('folder', 'set', 'folder.')
flags.DEFINE_string('root', None, 'root folder.')
flags.DEFINE_bool('date', True, 'use date.')
flags.DEFINE_string('search', '', 'which search to use.')
flags.DEFINE_string('spaces', 'spaces', 'which search to use.')
flags.DEFINE_string('terminal', 'output_to_files', 'terminal for launchpad.')
flags.DEFINE_integer('idx', None, 'number of gpus per job. accepts fractions.')
flags.DEFINE_integer('skip', 1, 'skip run jobs.')
flags.DEFINE_integer('ray', 0, 'whether to use ray tune.')
flags.DEFINE_integer('debug_search', 0, 'whether to use ray tune.')


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
  space = importlib.import_module(f'projects.kitchen_combo.{FLAGS.spaces}').get(FLAGS.search, FLAGS.agent)

  if FLAGS.idx is not None:
    listify_space(space)
    if FLAGS.idx < len(space):
      space = space[FLAGS.idx]
    else:
      return

  # root_path is needed to tell program absolute path
  # this is used for BabyAI
  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder if FLAGS.folder else "results/kitchen_combo/refactor"
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

  default_env_kwargs=dict(setting='medium')
  run_experiments(
    build_program_fn=build_program,
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
