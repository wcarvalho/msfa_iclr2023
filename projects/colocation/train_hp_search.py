"""
Param search.
"""

""" Train command:

 PYTHONPATH=$PYTHONPATH:$HOME/successor_features/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/colocation/train_hp_search.py \
    --search usfa_comparison_oneseed --num_gpus 4 --wandb_name usfa_showdown
    
"""


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

from projects.colocation.train_distributed import build_program

flags.DEFINE_string('root_path','/home/nameer/successor_features/rljax','path to run program from')
flags.DEFINE_string('search', 'usfa_comparison_oneseed', 'which search to use.')
flags.DEFINE_string('spaces', 'search', 'which search to use.')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus per job. accepts fractions.')
#all other flags are defined in train_distributed


FLAGS = flags.FLAGS

def main(_):
  mp.set_start_method('spawn')
  num_cpus = 3
  num_gpus = FLAGS.num_gpus
  DEFAULT_NUM_ACTORS = 4
  DEFAULT_SIMPLE = False
  DEFAULT_AGENT = 'r2d1_noise'
  DEFAULT_NOWALLS = False
  DEFAULT_ONE_ROOM = False
  DEFAULT_DETERMINISTIC_ROOMS = True
  DEFAULT_ROOM_REWARD = 0.

  space = importlib.import_module(f'projects.colocation.{FLAGS.spaces}').get(FLAGS.search)

  use_wandb = FLAGS.wandb
  wandb_project = FLAGS.wandb_project
  wandb_entity = FLAGS.wandb_entity
  wandb_notes = FLAGS.wandb_notes
  wandb_name = FLAGS.wandb_name
  root_path = FLAGS.root_path

  def create_and_run_program(config):
    """Create and run launchpad program
    """
    agent = config.pop('agent', DEFAULT_AGENT)
    num_actors = config.pop('num_actors', DEFAULT_NUM_ACTORS)
    simple = config.pop('simple',DEFAULT_SIMPLE)
    nowalls = config.pop('nowalls',DEFAULT_NOWALLS)
    one_room = config.pop('one_room',DEFAULT_ONE_ROOM)
    deterministic_rooms = config.pop('deterministic_rooms',DEFAULT_DETERMINISTIC_ROOMS)
    room_reward = config.pop('room_reward',DEFAULT_ROOM_REWARD)


    # -----------------------
    # wandb settings
    # -----------------------
    wandb_init_kwargs = dict(
      project=wandb_project,
      entity=wandb_entity,
      group=agent,  # overall group
      notes=wandb_notes,
      config=dict(space=space),
      save_code=True,
    )

    os.chdir(root_path)
    # -----------------------
    # launch experiment
    # -----------------------
    program = build_program(
      agent=agent,
      num_actors=num_actors,
      config_kwargs=config,
      wandb_init_kwargs=wandb_init_kwargs if use_wandb else None,
      simple=simple,
      nowalls=nowalls,
      one_room=one_room,
      deterministic_rooms=deterministic_rooms,
      room_reward=room_reward,
      wandb_name=wandb_name
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

  if isinstance(space, dict):
    space = [space]
  elif isinstance(space, list):
    assert isinstance(space[0], dict)
  else:
    raise RuntimeError(type(space))

  experiment_specs = [tune.Experiment(
        name="goto",
        run=train_function,
        config=s,
        resources_per_trial={"cpu": num_cpus, "gpu": num_gpus},
        local_dir='~/successor_features/tmp/ray',
      ) for s in space
  ]
  all_trials = tune.run_experiments(experiment_specs)


if __name__ == '__main__':
  app.run(main)