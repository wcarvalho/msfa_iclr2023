"""
Param search.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES="0,1,2,3" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/msf/goto_search.py \
    --folder 'results/msf/final/goto_avoid' \
    --date=False \
    --search baselines_small \
    --num_gpus 1

  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/msf/goto_search.py \
    --folder 'results/msf/refactor' \
    --search usfa_lstm
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

from projects.msf.goto_distributed import build_program

flags.DEFINE_string('folder', 'set', 'folder.')
flags.DEFINE_string('root', None, 'root folder.')
flags.DEFINE_bool('date', True, 'use date.')
flags.DEFINE_string('search', 'baselines', 'which search to use.')
flags.DEFINE_float('num_gpus', 1, 'number of gpus per job. accepts fractions.')

FLAGS = flags.FLAGS

def main(_):
  mp.set_start_method('spawn')
  experiment=None
  num_cpus = 3
  num_gpus = FLAGS.num_gpus
  DEFAULT_ENV_SETTING = 'large_respawn'
  DEFAULT_NUM_ACTORS = 4

  search = FLAGS.search
  if search == 'baselines':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4, 5]),
        "agent": tune.grid_search(
          ['usfa', 'r2d1', 'r2d1_noise_eval']),
        "setting": tune.grid_search(['large_respawn']),
        "importance_sampling_exponent": tune.grid_search([0.0]),
    }
    experiment='baselines'
  elif search == 'usfa_lstm':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa_lstm']),
        "normalize_cumulants": tune.grid_search([False]),
        "delta_cumulant": tune.grid_search([False]),
        "reward_loss": tune.grid_search(['l2']),
        "reward_coeff": tune.grid_search([1e-1, 1e-2]),
        "value_coeff": tune.grid_search([0, 1.0]),
    }
    experiment='fixed_env_1'
  elif search == 'usfa_farm_qlearning':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa_farmflat_qlearning']),
        "module_attn_heads": tune.grid_search([0, 4]),
        "shared_module_attn": tune.grid_search([True, False]),

    }
    experiment='usfa_farm_q1'
  elif search == 'usfa_farm_nomodel':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa_farmflat']),
        "model_coeff": tune.grid_search([0.]),
        "value_coeff": tune.grid_search([1.]), # Q-learning
        "loss_coeff": tune.grid_search([1e-1, 1e-2, 1e-3]), # SF
        "reward_coeff": tune.grid_search([1e-1, 1e-2, 1e-3]), # reward coeff
        "delta_cumulant": tune.grid_search([False]),

    }
    experiment='no_model'
  elif search == 'usfa_farm':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa_farmflat']),
        # "model_coeff": tune.grid_search([1e-1, 1e-2]),
        # "value_coeff": tune.grid_search([100., 1000.0]),
        # "reward_coeff": tune.grid_search([1.]),
        # "loss_coeff": tune.grid_search([1e-1, 1e-2]),
        "model_coeff": tune.grid_search([0.]),
        "value_coeff": tune.grid_search([100., 1000.0]),
        "reward_coeff": tune.grid_search([0.]),
        "loss_coeff": tune.grid_search([0.]),

    }
    experiment='no_model'
  else:
    raise NotImplementedError(search)



  # root_path is needed to tell program absolute path
  # this is used for BabyAI

  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder if FLAGS.folder else "results/msf/refactor"
  use_date = FLAGS.date

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', 'r2d1')
    num_actors = config.pop('num_actors', DEFAULT_NUM_ACTORS)
    setting = config.pop('setting', DEFAULT_ENV_SETTING)


    # get log dir for experiment
    log_dir = gen_log_dir(
      base_dir=os.path.join(root_path,folder),
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

    os.chdir(root_path)
    # launch experiment
    program = build_program(
      agent=agent, num_actors=num_actors,
      use_wandb=False,
      setting=setting,
      config_kwargs=config, 
      path=root_path,
      log_dir=log_dir)

    if program is None: return
    controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal='current_terminal', 
      local_resources = { # minimize GPU footprint
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))
          }
      )
    time.sleep(60) # sleep for 15 seconds
    controller.wait()



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
