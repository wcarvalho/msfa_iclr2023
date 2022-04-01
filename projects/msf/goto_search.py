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
from pprint import pprint
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
  DEFAULT_NUM_ACTORS = 8
  name_kwargs=[]

  search = FLAGS.search
  if search == 'agent':
    space = {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([FLAGS.agent]),
        "reward_coeff": tune.grid_search([1e-4]),
        "out_layers" : tune.grid_search([0]),
    }
    experiment='baselines'
    name_kwargs=[]
  elif search == 'baselines':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(
          ['usfa', 'r2d1', 'r2d1_noise_eval', 'r2d1_no_task', 'usfa_lstm', 'r2d1_farm']),
        "setting": tune.grid_search(['large_respawn']),
        "importance_sampling_exponent": tune.grid_search([0.0]),
    }
    experiment='baselines'
    name_kwargs=[]
  elif search == 'baselines_norespawn':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(
          ['usfa', 'r2d1', 'r2d1_noise_eval']),
        "setting": tune.grid_search(['large']),
    }
    experiment='baselines_norespawn'
    name_kwargs=[]
  elif search == 'usfa_unsup':
    space = {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(
          ['usfa_farm_model', 'usfa_farmflat_model']),
        "setting": tune.grid_search(['large_respawn']),
        "q_aux": tune.grid_search(['single']),
        "loss_coeff": tune.grid_search([1]),
        "num_sgd_steps_per_step": tune.grid_search([4])
    }
    experiment='baselines'
    name_kwargs=[]
  elif search == 'usfa_farm_speed':
    space = {
        "agent": tune.grid_search(['usfa', 'r2d1', 'usfa_farm']),
        "seed": tune.grid_search([1]),
        # "image_attn" : tune.grid_search([True]),
        # "nmodules" : tune.grid_search([4]),
        "samples_per_insert" : tune.grid_search([6.0, 8.0]),
        # "farm_vmap" : tune.grid_search(["lift"]),
    }

  elif search == 'usfa_farm_model_v1':
    space = {
        "agent": tune.grid_search(['usfa_farm_model']),
        "seed": tune.grid_search([1,2,3]),
        "reward_coeff": tune.grid_search([1e-4]),
        "max_number_of_steps" : tune.grid_search([3_000_000]),
        "setting": tune.grid_search(["small"]),
        "seperate_cumulant_params" : tune.grid_search([True]),
        "seperate_model_params" : tune.grid_search([False]),
        "seperate_value_params" : tune.grid_search([True]),
    }
    name_kwargs=[
      "seperate_cumulant_params",
      "seperate_model_params",
      "seperate_value_params",
    ]
  elif search == 'usfa_farm_model_v2':
    space = {
        "agent": tune.grid_search(['usfa_farm_model']),
        "seed": tune.grid_search([1,2,3]),
        "reward_coeff": tune.grid_search([1e-4]),
        "max_number_of_steps" : tune.grid_search([3_000_000]),
        "setting": tune.grid_search(["small"]),
        "seperate_cumulant_params" : tune.grid_search([False]),
        "seperate_model_params" : tune.grid_search([True]),
        "seperate_value_params" : tune.grid_search([False]),
    }
    name_kwargs=[
      "seperate_cumulant_params",
      "seperate_model_params",
      "seperate_value_params",
    ]
  elif search == 'usfa_farm_model_v3':
    space = {
        "agent": tune.grid_search(['usfa_farm_model']),
        "seed": tune.grid_search([1,2,3]),
        "reward_coeff": tune.grid_search([1e-4]),
        "max_number_of_steps" : tune.grid_search([3_000_000]),
        "setting": tune.grid_search(["small"]),
        "seperate_cumulant_params" : tune.grid_search([False]),
        "seperate_model_params" : tune.grid_search([True]),
        "seperate_value_params" : tune.grid_search([True]),
    }
    name_kwargs=[
      "seperate_cumulant_params",
      "seperate_model_params",
      "seperate_value_params",
    ]
  elif search == 'usfa_farm_model_v4':
    space = {
        "agent": tune.grid_search(['usfa_farm_model']),
        "seed": tune.grid_search([1,2,3]),
        "reward_coeff": tune.grid_search([1e-4]),
        "max_number_of_steps" : tune.grid_search([3_000_000]),
        "setting": tune.grid_search(["small"]),
        "seperate_cumulant_params" : tune.grid_search([True]),
        "seperate_model_params" : tune.grid_search([False]),
        "seperate_value_params" : tune.grid_search([False]),
    }
    name_kwargs=[
      "seperate_cumulant_params",
      "seperate_model_params",
      "seperate_value_params",
    ]
  else:
    raise NotImplementedError(search)

  # root_path is needed to tell program absolute path
  # this is used for BabyAI

  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder if FLAGS.folder else "results/msf/refactor"
  use_date = FLAGS.date
  use_wandb = FLAGS.wandb
  group = FLAGS.group # overall group
  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=FLAGS.group, # overall group
    notes=FLAGS.wandb_notes,
    config=dict(space=space),
  )

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', 'r2d1')
    num_actors = config.pop('num_actors', DEFAULT_NUM_ACTORS)
    setting = config.pop('setting', DEFAULT_ENV_SETTING)

    # -----------------------
    # get log dir for experiment
    # -----------------------
    log_path_config=dict(
      agent=agent,
      setting=setting,
      **({'exp': experiment} if experiment else {}),
      **config
      )
    log_dir, config_path_str = gen_log_dir(
      base_dir=os.path.join(root_path, folder, group),
      hourminute=False,
      return_kwpath=True,
      date=use_date,
      **log_path_config
      )

    print("="*50)
    if not os.path.exists(log_dir):
      print(f"RUNNING\n{log_dir}")
      print("="*50)
    else:
      print(f"SKIPPING\n{log_dir}")
      print("="*50)
      return

    # -----------------------
    # wandb settings
    # -----------------------
    if name_kwargs:
      try:
        name = [f'{k}={log_path_config[k]}' for k in name_kwargs]
        name = ','.join(name)
      except Exception as e:
        print(e)
        name = config_path_str
    else:
      name = config_path_str
    wandb_init_kwargs['name']=name # short display name for run

    # needed for various services (wandb, etc.)
    os.chdir(root_path)

    # -----------------------
    # launch experiment
    # -----------------------
    program = build_program(
      agent=agent,
      num_actors=num_actors,
      config_kwargs=config, 
      wandb_init_kwargs=wandb_init_kwargs if use_wandb else None,
      setting=setting,
      path=root_path,
      log_dir=log_dir,
      )

    if program is None: return
    controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal='current_terminal', 
      local_resources = { # minimize GPU footprint
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))
          }
      )
    time.sleep(60) # sleep for 60 seconds to avoid collisions
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
