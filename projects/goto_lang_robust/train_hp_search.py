"""
Param search.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/goto_lang_robust/train_hp_search.py \
    --search r2d1_search \
    --folder 'results/msf/refactor/goto_language_test1'

  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES="5,6,7" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/goto_lang_robust/train_hp_search.py \
    --folder 'results/msf/final/goto_language_test1' \
    --search baselines \
    --date=False
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

from utils import gen_log_dir
import os

from projects.goto_lang_robust.train_distributed import build_program

flags.DEFINE_string('folder', 'set', 'folder.')
flags.DEFINE_string('root', None, 'root folder.')
flags.DEFINE_bool('date', True, 'use date.')
flags.DEFINE_string('search', 'model', 'root folder.')

FLAGS = flags.FLAGS

def main(_):
  mp.set_start_method('spawn')
  experiment=None
  num_cpus = 6
  num_gpus = 1

  search = FLAGS.search
  if search == 'baselines':
    space = {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1', 'r2d1_noise_ensemble']),
        "setting": tune.grid_search([1, 2]),
    }
    experiment='baselines'
  elif search == 'r2d1_search':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1']),
        # "word_dim": tune.grid_search([32, 64, 128]),
        # "word_initializer": tune.grid_search(["RandomNormal", "TruncatedNormal"]),
        # "vision_torso": tune.grid_search(['babyai']),
        # "num_epsilons": tune.grid_search([128, 256]),
        "room_size": tune.grid_search([5]),
        "num_dists": tune.grid_search([1]),
        "instr": tune.grid_search(["goto", "pickup"]),
        "task_in_memory": tune.grid_search(["goto", "pickup"]),
        "max_replay_size": tune.grid_search([100_000, 200_000, 400_000]),
    }
    experiment='r2d1_babyai'
  else:
    raise NotImplementedError



  # root_path is needed to tell program absolute path
  # this is used for BabyAI

  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder
  use_date = FLAGS.date

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', 'r2d1')
    num_actors = config.pop('num_actors', 9)
    setting = config.pop('setting', 1)


    # get log dir for experiment
    log_dir = gen_log_dir(
      base_dir=os.path.join(root_path, folder),
      hourminute=False,
      date=use_date,
      agent=agent,
      setting=setting,
      **({'exp': experiment} if experiment else {}),
      **config)

    if not os.path.exists(log_dir):
      print("="*50)
      print(f"RUNNING\n{log_dir}")
      print("="*50)
    else:
      print("="*50)
      print(f"SKIPPING\n{log_dir}")
      print("="*50)
      return

    # launch experiment
    program = build_program(
      agent=agent, num_actors=num_actors,
      use_wandb=True,
      setting=setting,
      config_kwargs=config, 
      path=root_path,
      log_dir=log_dir)

    if program is None: return
    lp.launch(program, lp.LaunchType.LOCAL_MULTI_THREADING, terminal='current_terminal', 
      # local_resources = { # minimize GPU footprint
      # 'actor':
      #     PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      # 'evaluator':
      #     PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))
      #     }
      )
    time.sleep(30) # sleep for 15 seconds



  def train_function(config):
    """Run inside threads and creates new process.
    """
    p = mp.Process(
      target=create_and_run_program, 
      args=(config,))
    p.start()
    p.join() # this blocks until the process terminates
    # this will call right away and end.



  experiment_spec = tune.Experiment(
      name="goto",
      run=train_function,
      config=space,
      resources_per_trial={"cpu": num_cpus, "gpu": num_gpus}, 
      local_dir='/tmp/ray',
    )
  all_trials = tune.run_experiments(experiment_spec)


if __name__ == '__main__':
  app.run(main)
