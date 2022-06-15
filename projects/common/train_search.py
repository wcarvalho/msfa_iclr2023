from absl import app
from absl import flags
from pathlib import Path
from hyperopt import hp
import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess
# from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import ParameterGrid

import functools
from ray import tune
import multiprocessing as mp
import jax
import time
from pprint import pprint
from utils import gen_log_dir
import os
import importlib
from pprint import pprint 
import sklearn
import jax

DEFAULT_NUM_ACTORS=3
DEFAULT_LABEL=''

def create_and_run_program(config, build_program_fn, root_path, folder, group, wandb_init_kwargs, default_env_kwargs=None, use_wandb=True, terminal='current_terminal', skip=True, debug=False):
  """Create and run launchpad program
  """

  agent = config.pop('agent', 'r2d1')
  num_actors = config.pop('num_actors', DEFAULT_NUM_ACTORS)
  cuda = config.pop('cuda', None)
  group = config.pop('group', group)
  label = config.pop('label', DEFAULT_LABEL)

  if cuda:
    os.environ['CUDA_VISIBLE_DEVICES']=str(cuda)
  # -----------------------
  # add env kwargs to path desc
  # -----------------------
  default_env_kwargs = default_env_kwargs or {}

  env_kwargs = dict()
  for key, value in default_env_kwargs.items():
    env_kwargs[key] = config.pop(key, value)

  # only use non-default
  env_path=dict()
  for k,v in env_kwargs.items():
    if v != default_env_kwargs[k]:
      env_path[k]=v
  if label:
    env_path['L']=label
  # -----------------------
  # get log dir for experiment
  # -----------------------
  log_path_config=dict(
    agent=agent,
    **env_path,
    **config
    )
  log_dir, config_path_str = gen_log_dir(
    base_dir=os.path.join(root_path, folder, group),
    hourminute=False,
    return_kwpath=True,
    date=False,
    path_skip=['max_number_of_steps'],
    **log_path_config
    )

  # os.environ['LAUNCHPAD_LOGGING_DIR']=log_dir
  print("="*50)
  if os.path.exists(log_dir) and skip:
    print(f"SKIPPING\n{log_dir}")
    print("="*50)
    return
  else:
    print(f"RUNNING\n{log_dir}")
    print("="*50)


  # -----------------------
  # wandb settings
  # -----------------------
  name = config_path_str
  if wandb_init_kwargs:
    wandb_init_kwargs['name']=name # short display name for run
    if group is not None:
      wandb_init_kwargs['group']=group # short display name for run

  # needed for various services (wandb, etc.)
  os.chdir(root_path)

  # -----------------------
  # launch experiment
  # -----------------------
  agent = build_program_fn(
    agent=agent,
    num_actors=num_actors,
    config_kwargs=config, 
    wandb_init_kwargs=wandb_init_kwargs if use_wandb else None,
    env_kwargs=env_kwargs,
    path=root_path,
    log_dir=log_dir,
    build=False,
    )
  program = agent.build()

  local_resources = {
      "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
  }
  if cuda:
    local_resources['learner'] = PythonProcess(
      env={"CUDA_VISIBLE_DEVICES": str(cuda)})

  if debug:
    return

  controller = lp.launch(program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal=terminal, 
    local_resources=local_resources
    )
  controller.wait()
  print("Controller finished")
  if agent.wandb_obj:
    agent.wandb_obj.finish()
  time.sleep(120) # sleep for 60 seconds to avoid collisions


def manual_parallel(fn, space, debug=False):
  """Run in parallel manually."""
  
  configs = []
  gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
  idx = 0
  for x in space:
    y = {k:list(v.values())[0] for k,v in x.items()}
    grid = list(sklearn.model_selection.ParameterGrid(y))

    # assign gpus
    for g in grid:
      g['cuda'] = gpus[idx%len(gpus)]
      idx += 1
    configs.extend(grid)

  pprint(configs)
  print('nconfigs', len(configs))
  print('gpus', len(gpus))

  idx = 1
  processes = []

  for config in configs:
    wait = idx % len(gpus) == 0

    p = mp.Process(
      target=fn,
      args=(config,))
    p.start()
    processes.append(p)
    if not debug:
      time.sleep(60) # sleep for 60 seconds to avoid collisions
    if wait:
      print("="*100)
      print("Waiting")
      print("="*100)
      for p in processes:
        p.join() # this blocks until the process terminates
      processes = []
      print("="*100)
      print("Running new set")
      print("="*100)
    idx += 1
    if not debug:
      time.sleep(120) # sleep for 120 seconds to finish syncing++

def listify_space(space):
  if isinstance(space, dict):
    space = [space]
  elif isinstance(space, list):
    assert isinstance(space[0], dict)
  else:
    raise RuntimeError(type(space))
  return space

def run_experiments(
  build_program_fn,
  space,
  root_path,
  folder,
  group,
  wandb_init_kwargs,
  default_env_kwargs=None,
  use_wandb=True,
  terminal='current_terminal',
  num_cpus=4,
  num_gpus=1,
  skip=True,
  use_ray=False,
  debug=False):
  
  space = listify_space(space)
  if debug:
    print("="*30)
    print("DEBUGGING")
    print("="*30)

  if use_ray:
    def train_function(config):
      """Run inside threads and creates new process.
      """
      p = mp.Process(
        target=create_and_run_program, 
        args=(config,),
        kwargs=dict(
          build_program_fn=build_program_fn,
          root_path=root_path,
          folder=folder,
          group=group,
          wandb_init_kwargs=wandb_init_kwargs,
          default_env_kwargs=default_env_kwargs,
          use_wandb=use_wandb,
          terminal=terminal,
          skip=skip)
        )
      p.start()
      if not debug:
        time.sleep(60)
      p.join() # this blocks until the process terminates
      # this will call right away and end.

    experiment_specs = [tune.Experiment(
          name="goto",
          run=train_function,
          config=s,
          resources_per_trial={"cpu": num_cpus, "gpu": num_gpus}, 
          local_dir='/tmp/ray',
        ) for s in space
    ]
    all_trials = tune.run_experiments(experiment_specs)

  else:
    manual_parallel(
      fn=functools.partial(create_and_run_program,
        build_program_fn=build_program_fn,
        root_path=root_path,
        folder=folder,
        group=group,
        wandb_init_kwargs=wandb_init_kwargs,
        default_env_kwargs=default_env_kwargs,
        use_wandb=use_wandb,
        terminal=terminal,
        debug=debug,
        skip=skip),
      space=space,
      debug=debug)