from absl import app
from absl import flags
from pathlib import Path
from hyperopt import hp
import launchpad as lp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
import multiprocessing as mp


from utils import gen_log_dir
import os

from projects.msf.goto_distributed import build_program

flags.DEFINE_string('experiment', 'experiment', 'experiment_name.')

FLAGS = flags.FLAGS

def main(_):
  mp.set_start_method('spawn')

  space = {
      # "seed": tune.grid_search([1, 2, 3]),
      "agent": tune.grid_search(['r2d1', 'usfa']),
      "seed": tune.grid_search([1, 2]),
      # "agent": tune.grid_search(['r2d1']),
  }
  num_cpus = 1
  num_gpus = 1

  # root_path is needed to tell program absolute path
  # this is used for BabyAI
  root_path = str(Path().absolute()) 
  experiment_name=FLAGS.experiment

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', 'r2d1')
    num_actors = config.pop('num_actors', 10)


    # get log dir for experiment
    log_dir = gen_log_dir(
      base_dir=f"{root_path}/results/msf/search",
      hourminute=False,
      agent=agent,
      **config)
    print("="*50)
    print(f"RUNNING\n{log_dir}")
    print("="*50)

    # launch experiment
    program = build_program(agent, num_actors,
      config_kwargs=config, 
      path=root_path,
      log_dir=log_dir)
    lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal='current_terminal')



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
