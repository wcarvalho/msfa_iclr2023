"""
Param search.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES="1,2,3" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/msf/goto_search.py \
    --folder 'reward'
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

FLAGS = flags.FLAGS

def main(_):
  mp.set_start_method('spawn')
  experiment=None

  search = 'baselines'
  if search == 'baselines':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'r2d1_farm', 'usfa']),
    }
    # experiment='r2d1_farm_model_v1'
  elif search == 'r2d1_farm':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1_farm_model']),
        "out_layers": tune.grid_search( [1, 2]),
        # "latent_source": tune.grid_search( ["samples", "memory"]),
        "model_layers": tune.grid_search( [1, 2]),
    }
    # experiment='r2d1_farm_model_v1'
  elif search == 'r2d1_vae':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1_vae']),
        "vae_coeff": tune.grid_search( [1e-3, 1e-4]),
        "beta": tune.grid_search( [25, 100]),
        # "latent_source": tune.grid_search( ["samples", "memory"]),
        "latent_dim": tune.grid_search( [512]),
    }
    experiment='vae_beta_v5'
  elif search == 'usfa':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa']),
    }
  elif search == 'usfa_farm':
    # space = {
    #     "seed": tune.grid_search([1]),
    #     # "agent": tune.grid_search(['usfa_farm_model', 'usfa_farmflat_model']),
    #     "agent": tune.grid_search(['usfa_farmflat_model']),
    #     # "out_layers": tune.grid_search([2]),
    #     # "extra_negatives": tune.grid_search([0, 10]),
    #     # "normalize_task": tune.grid_search([False]),
    #     "normalize_cumulants": tune.grid_search([True]),
    #     "reward_loss": tune.grid_search(['l2']),
    #     # "model_coeff": tune.grid_search([10, 100]),
    #     "reward_coeff": tune.grid_search([100]),
    #     "value_coeff": tune.grid_search([1, 100]),
    # }
    space = {
        "seed": tune.grid_search([1]),
        # "agent": tune.grid_search(['usfa_farm_model', 'usfa_farmflat_model']),
        "agent": tune.grid_search(['usfa_farmflat_model']),
        # "out_layers": tune.grid_search([2]),
        # "extra_negatives": tune.grid_search([0, 10]),
        # "normalize_task": tune.grid_search([False]),
        "normalize_cumulants": tune.grid_search([False]),
        "reward_loss": tune.grid_search(['l2', 'binary']),
        "model_coeff": tune.grid_search([1, 10]),
        "reward_coeff": tune.grid_search([1e-1]),
        "value_coeff": tune.grid_search([1]),
    }
  else:
    raise NotImplementedError


  num_cpus = 2
  num_gpus = 1

  # root_path is needed to tell program absolute path
  # this is used for BabyAI

  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', 'r2d1')
    num_actors = config.pop('num_actors', 6)
    setting = config.pop('setting', 'small')


    # get log dir for experiment
    log_dir = gen_log_dir(
      base_dir=f"{root_path}/results/msf/{folder}",
      hourminute=False,
      agent=agent,
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
    lp.launch(program, lp.LaunchType.LOCAL_MULTI_THREADING, terminal='current_terminal', 
      local_resources = { # minimize GPU footprint
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))
          }
      )
    time.sleep(15) # sleep for 15 seconds



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
