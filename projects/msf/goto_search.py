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

from utils import gen_log_dir
import os

from projects.msf.goto_distributed import build_program

flags.DEFINE_string('folder', 'set', 'folder.')

FLAGS = flags.FLAGS

def main(_):
  mp.set_start_method('spawn')

  space = {
      "seed": tune.grid_search([1]),
      "agent": tune.grid_search(['usfa_reward']),
      "reward_coeff": tune.grid_search([1, .01]),
      "reward_loss": tune.grid_search(['l2', 'binary']),
  }
  # space = {
  #     "seed": tune.grid_search([1]),
  #     "agent": tune.grid_search(['usfa']),
  # }
  experiment='check'
  # space = ParameterGrid(space.values())
  # space = [p for p in space]
  # space = jax.tree_map(lambda x: tune.grid_search([x]), space)

  num_cpus = 1
  num_gpus = 1

  # root_path is needed to tell program absolute path
  # this is used for BabyAI
  root_path = str(Path().absolute()) 
  folder=FLAGS.folder

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', 'r2d1')
    num_actors = config.pop('num_actors', 10)


    # get log dir for experiment
    log_dir = gen_log_dir(
      base_dir=f"{root_path}/results/msf/{folder}",
      hourminute=False,
      agent=agent,
      **config)
    print("="*50)
    print(f"RUNNING\n{log_dir}")
    print("="*50)

    # launch experiment
    program = build_program(agent, num_actors,
      experiment=experiment,
      config_kwargs=config, 
      path=root_path,
      log_dir=log_dir)
    lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal='current_terminal', 
      local_resources = { # minimize GPU footprint
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))
          }
      )



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
