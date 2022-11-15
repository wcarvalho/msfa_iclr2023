"""
Param search.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import functools
from absl import app
from absl import flags
from pathlib import Path
from hyperopt import hp
import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess
# from raytune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import ParameterGrid
from ray import tune
import multiprocessing as mp
import jax
import time
from pprint import pprint
from utils import gen_log_dir
import importlib

from experiments.common.train_search import run_experiments, listify_space

from experiments.exploration2.train_distributed import build_program

# flags.DEFINE_string('folder', 'set', 'folder.')
# flags.DEFINE_string('root', None, 'root folder.')
# flags.DEFINE_bool('date', True, 'use date.')
# flags.DEFINE_string('search', '', 'which search to use.')
# flags.DEFINE_string('spaces', 'brain_search', 'which search to use.')
# flags.DEFINE_string('terminal', 'output_to_files', 'terminal for launchpad.')
# flags.DEFINE_string('actor_label', None, '.')
# flags.DEFINE_string('evaluator_label', None, '.')
# flags.DEFINE_float('num_gpus', 1, 'number of gpus per job. accepts fractions.')
# flags.DEFINE_integer('num_cpus', 3, 'number of gpus per job. accepts fractions.')
# flags.DEFINE_integer('actors', 4, 'number of gpus per job. accepts fractions.')
# flags.DEFINE_integer('skip', 1, 'skip run jobs.')
# flags.DEFINE_integer('ray', 0, 'whether to use ray tune.')
# flags.DEFINE_integer('idx', None, 'number of gpus per job. accepts fractions.')

def main(_):
  FLAGS = flags.FLAGS
  terminal = FLAGS.terminal
  mp.set_start_method('spawn')
  num_cpus = int(FLAGS.num_cpus)
  num_gpus = float(FLAGS.num_gpus)

  assert FLAGS.search != '', 'set search!'
  space, actor_label, evaluator_label = importlib.import_module(f'experiments.exploration2.{FLAGS.spaces}').get(FLAGS.search, FLAGS.agent)
  if FLAGS.idx is not None:
    listify_space(space)
    if FLAGS.idx < len(space):
      space = space[FLAGS.idx]
    else:
      return


  # root_path is needed to tell program absolute path
  # this is used for BabyAI
  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder if FLAGS.folder else "results/exploration2/refactor"
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

  default_env_kwargs = {
    'setting' : 'multiv9',
    'task_reps' : 'object_verbose',
    'room_size' : 7,
    'num_dists' : 0,
    'symbolic' : False,
    'struct_and': False,
    'task_reset_behavior': 'none',
  }
  run_experiments(
    build_program_fn=build_program,
    space=space,
    root_path=root_path,
    folder=folder,
    group=group,
    wandb_init_kwargs=wandb_init_kwargs,
    default_env_kwargs=default_env_kwargs,
    use_wandb=use_wandb,
    build_kwargs=dict(
      actor_label=actor_label,
      evaluator_label=evaluator_label),
    terminal=FLAGS.terminal,
    skip=FLAGS.skip,
    use_ray=FLAGS.ray)



if __name__ == '__main__':
  app.run(main)
