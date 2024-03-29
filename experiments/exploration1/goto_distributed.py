"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/experiments/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/msfa/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python experiments/exploration1/goto_distributed.py \
    --agent r2d1
"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from absl import app
from absl import flags
import acme
from acme.utils import paths
import functools

from agents import td_agent
from utils import make_logger, gen_log_dir
from utils import data as data_utils

from experiments.exploration1 import helpers
from experiments.common.train_distributed import build_common_program
from experiments.common.observers import LevelReturnObserver, LevelAvgReturnObserver

# -----------------------
# flags
# -----------------------
# flags.DEFINE_string('agent', 'r2d1', 'which agent.')
# flags.DEFINE_integer('seed', 1, 'Random seed.')
# flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
# flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')
# # WANDB
# flags.DEFINE_bool('debug', False, 'whether to debug.')
# flags.DEFINE_bool('custom_loggers', True, 'whether to use custom loggers.')
# flags.DEFINE_bool('wandb', False, 'whether to log.')
# flags.DEFINE_string('wandb_project', 'msf2', 'wand project.')
# flags.DEFINE_string('wandb_entity', None, 'wandb entity')
# flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
# flags.DEFINE_string('notes', '', 'notes for wandb.')

FLAGS = flags.FLAGS

def build_program(
  agent: str,
  num_actors : int,
  save_config_dict: dict=None,
  wandb_init_kwargs=None,
  update_wandb_name=True, # use path from logdir to populate wandb name
  env_kwargs=None,
  group='experiments', # subdirectory that specifies experiment group
  hourminute=True, # whether to append hour-minute to logger path
  log_every=30.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  debug: bool=False,
  return_avg_episodes=200,
  **kwargs,
  ):
  env_kwargs = env_kwargs or dict()
  config_kwargs = config_kwargs or dict()
  if debug:
    config_kwargs['eval_task_support'] = 'eval'
    print("="*50)
    print("DEBUG")
    print("="*50)

  setting = env_kwargs.get('setting', 'large_respawn')
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment(
      evaluation=is_eval, path=path, setting=setting)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = helpers.load_agent_settings(agent, env_spec, config_kwargs, setting=setting)

  if debug:
      config.batch_size = 32
      config.burn_in_length = 0
      config.trace_length = 20 # shorter
      config.sequence_period = 40
      config.prefetch_size = 0
      config.samples_per_insert_tolerance_rate = 0.1
      config.samples_per_insert = 0.0 # different
      config.num_parallel_calls = 1
      config.min_replay_size = 1_000 # smaller
      config.max_replay_size = 10_000 # smaller
      kwargs['colocate_learner_replay'] = False

  # -----------------------
  # define dict to save. add some extra stuff here
  # -----------------------
  save_config_dict = save_config_dict or dict()
  save_config_dict.update(config.__dict__)
  save_config_dict.update(
    agent=agent,
    setting=setting,
    group=group
  )

  # -----------------------
  # data stuff:
  #   construct log directory if necessary
  #   + observer of data
  # -----------------------
  if not log_dir:
    log_dir, config_path_str = gen_log_dir(
      base_dir=f"{path}/results/exploration1/distributed/{group}",
      hourminute=hourminute,
      return_kwpath=True,
      seed=config.seed,
      agent=str(agent))

    if wandb_init_kwargs and update_wandb_name:
      wandb_init_kwargs['name'] = config_path_str

  observers = [LevelAvgReturnObserver(reset=return_avg_episodes)]
  # -----------------------
  # wandb settup
  # -----------------------
  os.chdir(path)
  return build_common_program(
    environment_factory=environment_factory,
    env_spec=env_spec,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    config=config,
    NetworkCls=NetworkCls,
    NetKwargs=NetKwargs,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    loss_label='Loss',
    num_actors=num_actors,
    save_config_dict=save_config_dict,
    log_every=log_every,
    log_with_key='log_data',
    observers=observers,
    **kwargs,
    )

def main(_):
  config_kwargs=dict(seed=FLAGS.seed)

  if FLAGS.max_number_of_steps is not None:
    config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps

  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=FLAGS.group if FLAGS.group else FLAGS.agent, # organize individual runs into larger experiment
    notes=FLAGS.notes,
  )

  program = build_program(
    agent=FLAGS.agent,
    num_actors=FLAGS.num_actors,
    config_kwargs=config_kwargs,
    wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
    debug=FLAGS.debug,
    # custom_loggers=FLAGS.custom_loggers,
    )

  # Launch experiment.
  controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='-1')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='-1'))}
  )
  controller.wait()

if __name__ == '__main__':
  app.run(main)
