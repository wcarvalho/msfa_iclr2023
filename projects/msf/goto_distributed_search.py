from pathlib import Path
from hyperopt import hp
import launchpad as lp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune

import os

from projects.msf.goto_distributed import build_program


space = {
    "seed": tune.grid_search([1, 2, 3]),
    "agent": tune.grid_search(['r2d1', 'usfa'])
    # "seed": tune.grid_search([1]),
    # "agent": tune.grid_search(['r2d1'])
}
num_cpus = 3
num_gpus = 1


root_path = str(Path().absolute())
base_dir = os.path.join(root_path, )
def train_function(config):
  agent = config.pop('agent', 'r2d1')
  num_actors = config.pop('num_actors', 10)
  program = build_program(agent, num_actors, 
    config_kwargs=config, 
    path=root_path, log_path="results/msf/search")
  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal='current_terminal')



experiment_spec = tune.Experiment(
    name="goto",
    run=train_function,
    config=space,
    resources_per_trial={"cpu": num_cpus, "gpu": num_gpus}, 
    local_dir='/tmp/ray',
  )
all_trials = tune.run_experiments(experiment_spec)
