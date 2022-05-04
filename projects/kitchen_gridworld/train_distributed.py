"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/msf/goto_distributed.py \
    --agent r2d1
"""

# Do not preallocate GPU memory for JAX.
import os

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

from projects.kitchen_gridworld import helpers
from projects.common.train_distributed import build_common_program
from projects.common.observers import LevelAvgReturnObserver

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')

# -----------------------
# WANDB
# -----------------------
flags.DEFINE_bool('debug', False, 'whether to debug.')
flags.DEFINE_bool('custom_loggers', True, 'whether to use custom loggers.')
flags.DEFINE_bool('wandb', False, 'whether to log.')
flags.DEFINE_string('wandb_project', 'kitchen_grid_dist', 'wand project.')
flags.DEFINE_string('wandb_entity', 'wcarvalho92', 'wandb entity')
flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('notes', '', 'notes for wandb.')

FLAGS = flags.FLAGS

def build_program(
  agent: str,
  num_actors : int,
  wandb_init_kwargs=None,
  update_wandb_name=True, # use path from logdir to populate wandb name
  # setting='SmallL2NoDist',
  # task_reps='lesslang',
  # room_size=8,
  # num_dists=0,
  group='experiments', # subdirectory that specifies experiment group
  hourminute=True, # whether to append hour-minute to logger path
  log_every=5.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  debug: bool=False,
  env_kwargs=dict(),
  **kwargs,
  ):
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment(
    evaluation=is_eval,
    path=path,
    # setting=setting,
    **env_kwargs,
    )
  env = environment_factory(False)
  max_vocab_size = len(env.env.instr_preproc.vocab) # HACK
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(agent, env_spec, config_kwargs)

  if debug:
      config.batch_size = 32
      config.burn_in_length = 0
      config.trace_length = 20 # shorter
      config.sequence_period = 40
      config.prefetch_size = 0
      config.samples_per_insert_tolerance_rate = 0.1
      config.samples_per_insert = 6.0 # different
      config.num_parallel_calls = 1
      config.min_replay_size = 100 # smaller
      config.max_replay_size = 10_000 # smaller
      kwargs['colocate_learner_replay'] = False

  # -----------------------
  # define dict to save. add some extra stuff here
  # -----------------------
  save_config_dict = config.__dict__
  save_config_dict.update(
    agent=agent,
    # setting=setting,
    group=group,
    **env_kwargs,
  )

  # -----------------------
  # data stuff:
  #   construct log directory if necessary
  #   + observer of data
  # -----------------------
  if not log_dir:
    log_dir, config_path_str = gen_log_dir(
      base_dir=f"{path}/results/kitchen_gridworld/distributed/{group}",
      hourminute=hourminute,
      return_kwpath=True,
      seed=config.seed,
      agent=str(agent))

    if wandb_init_kwargs and update_wandb_name:
      wandb_init_kwargs['name'] = config_path_str

  observers = [LevelAvgReturnObserver()]
  # -----------------------
  # wandb settup
  # -----------------------
  os.chdir(path)
  setting = env_kwargs['setting']
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
    num_actors=num_actors,
    save_config_dict=save_config_dict,
    log_every=log_every,
    observers=observers,
    loss_label='Loss',
    actor_label=f"actor_{setting}",
    evaluator_label=f"evaluator_{setting}",
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
    custom_loggers=FLAGS.custom_loggers,
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
