"""
Param search.
"""

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
import os
import importlib

from projects.kitchen_gridworld.train_distributed import build_program

flags.DEFINE_string('folder', 'set', 'folder.')
flags.DEFINE_string('root', None, 'root folder.')
flags.DEFINE_bool('date', True, 'use date.')
flags.DEFINE_string('search', 'baselines', 'which search to use.')
flags.DEFINE_string('spaces', 'brain_search', 'which search to use.')
flags.DEFINE_string('terminal', 'output_to_files', 'terminal for launchpad.')
flags.DEFINE_float('num_gpus', 1, 'number of gpus per job. accepts fractions.')
flags.DEFINE_integer('num_cpus', 3, 'number of gpus per job. accepts fractions.')
flags.DEFINE_integer('actors', 5, 'number of gpus per job. accepts fractions.')
flags.DEFINE_integer('skip', 1, 'skip run jobs.')
flags.DEFINE_integer('ray', 0, 'whether to use ray tune.')

DEFAULT_ENV_SETTING = 'SmallL2NoDist'
DEFAULT_TASK_REPS='pickup'
DEFAULT_LABEL=''
DEFAULT_ROOM_SIZE=7
DEFAULT_NUM_ACTORS = 5
DEFAULT_NUM_DISTS = 0

def create_and_run_program(config, root_path, folder, group, wandb_init_kwargs, use_wandb, terminal, skip):
  """Create and run launchpad program
  """

  agent = config.pop('agent', 'r2d1')
  num_actors = config.pop('num_actors', DEFAULT_NUM_ACTORS)
  setting = config.pop('setting', DEFAULT_ENV_SETTING)
  task_reps = config.pop('task_reps', DEFAULT_TASK_REPS)
  room_size = config.pop('room_size', DEFAULT_ROOM_SIZE)
  num_dists = config.pop('num_dists', DEFAULT_NUM_DISTS)
  cuda = config.pop('cuda', None)
  group = config.pop('group', group)
  label = config.pop('label', DEFAULT_LABEL)

  if cuda:
    os.environ['CUDA_VISIBLE_DEVICES']=str(cuda)
  # -----------------------
  # add env kwargs to path desc
  # -----------------------
  default_env_kwargs = {
    'setting' : DEFAULT_ENV_SETTING,
    'task_reps' : DEFAULT_TASK_REPS,
    'room_size' : DEFAULT_ROOM_SIZE,
    'num_dists' : DEFAULT_NUM_DISTS,
  }

  env_kwargs=dict(
    setting=setting,
    task_reps=task_reps,
    room_size=room_size,
    num_dists=num_dists,
    )
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
  wandb_init_kwargs['name']=name # short display name for run
  if group is not None:
    wandb_init_kwargs['group']=group # short display name for run

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
    env_kwargs=env_kwargs,
    path=root_path,
    log_dir=log_dir,
    )

  local_resources = {
      "learner": PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(cuda)}
      ),
      "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
      "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
  }

  if program is None: return
  controller = lp.launch(program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal=terminal, 
    local_resources=local_resources
    )

  time.sleep(60) # sleep for 60 seconds to avoid collisions
  controller.wait()


def manual_parallel(fn, space):
  """Run in parallel manually."""
  from pprint import pprint 
  import sklearn
  import jax
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


  idx = 1
  processes = []
  for config in configs:
    wait = idx % len(gpus) == 0
    p = mp.Process(
      target=fn,
      args=(config,))
    p.start()
    processes.append(p)
    if wait:
      for p in processes:
        p.join() # this blocks until the process terminates
      processes = []
      print("="*50)
      print("Running new set")
      print("="*50)
    idx += 1



def main(_):
  FLAGS = flags.FLAGS
  terminal = FLAGS.terminal
  mp.set_start_method('spawn')
  num_cpus = int(FLAGS.num_cpus)
  num_gpus = float(FLAGS.num_gpus)

  space = importlib.import_module(f'projects.kitchen_gridworld.{FLAGS.spaces}').get(FLAGS.search, FLAGS.agent)
  if isinstance(space, dict):
    space = [space]
  elif isinstance(space, list):
    assert isinstance(space[0], dict)
  else:
    raise RuntimeError(type(space))

  # root_path is needed to tell program absolute path
  # this is used for BabyAI
  root_path = FLAGS.root if FLAGS.root else str(Path().absolute())
  folder=FLAGS.folder if FLAGS.folder else "results/kitchen_gridworld/refactor"
  use_date = FLAGS.date
  use_wandb = FLAGS.wandb
  group = FLAGS.group if FLAGS.group else FLAGS.search # overall group
  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=group, # overall group
    notes=FLAGS.notes,
    config=dict(space=space),
    save_code=True,
  )

  skip = FLAGS.skip

  if FLAGS.ray:
    def train_function(config):
      """Run inside threads and creates new process.
      """
      p = mp.Process(
        target=create_and_run_program, 
        args=(config,),
        kwargs=dict(
          root_path=root_path,
          folder=folder,
          group=group,
          wandb_init_kwargs=wandb_init_kwargs,
          use_wandb=use_wandb,
          terminal=terminal,
          skip=skip)
        )
      p.start()
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
    manual_parallel(fn=functools.partial(create_and_run_program,
      root_path=root_path,
      folder=folder,
      group=group,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=use_wandb,
      terminal=terminal,
      skip=skip),
    space=space)



if __name__ == '__main__':
  app.run(main)
