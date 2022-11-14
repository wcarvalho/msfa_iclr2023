"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/experiments/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
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

from experiments.exploration2 import helpers
from experiments.common.train_distributed import build_common_program
from experiments.common.observers import LevelAvgReturnObserver, LevelReturnObserver

# # -----------------------
# # flags
# # -----------------------
# flags.DEFINE_string('agent', 'r2d1', 'which agent.')
# flags.DEFINE_integer('seed', 1, 'Random seed.')
# flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
# flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')
# flags.DEFINE_string('env_setting', 'EasyPickup', 'which environment setting.')
# flags.DEFINE_string('task_reps', 'object_verbose', 'which task reps to use.')


# # -----------------------
# # WANDB
# # -----------------------
# flags.DEFINE_bool('debug', False, 'whether to debug.')
# flags.DEFINE_bool('custom_loggers', True, 'whether to use custom loggers.')
# flags.DEFINE_bool('wandb', False, 'whether to log.')
# flags.DEFINE_string('wandb_project', 'kitchen_grid_dist', 'wand project.')
# flags.DEFINE_string('wandb_entity', 'wcarvalho92', 'wandb entity')
# flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
# flags.DEFINE_string('name', '', 'same as wandb name. way to identify runs in a group.')
# flags.DEFINE_string('notes', '', 'notes for wandb.')

FLAGS = flags.FLAGS

def build_program(
  agent: str,
  num_actors : int,
  wandb_init_kwargs=None,
  update_wandb_name=True, # use path from logdir to populate wandb name
  group='experiments', # subdirectory that specifies experiment group
  hourminute=True, # whether to append hour-minute to logger path
  log_every=30.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  debug: bool=False,
  env_kwargs=None,
  actor_label=None,
  evaluator_label=None,
  save_config_dict=None,
  return_avg_episodes=200,
  **kwargs,
  ):
  config = config_kwargs or dict()
  env_kwargs = env_kwargs or dict(
    struct_and=True,
    task_reset_behavior='remove'
    )
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment(
    evaluation=is_eval,
    path=path,
    **env_kwargs,
    )
  env = environment_factory(False)
  max_vocab_size = max(env.env.instr_preproc.vocab.values())+1 # HACK
  tasks_file = env.tasks_file # HACK
  config['symbolic'] = env_kwargs.get('symbolic', False)
  # config_kwargs['step_penalty'] = env.step_penalty
  env_spec = acme.make_environment_spec(env)
  del env


  if debug:
    config['max_replay_size'] = 10_000
    config['min_replay_size'] = 100
    config['max_number_of_steps'] = 50_000
    config['learning_rate'] = 1e-2
    # config['cov_loss'] = 'l2_corr'
    config['cov_coeff'] = None
    config['max_gradient_norm'] = 1.0
    config['reward_coeff'] = 1.0
    # config['cov_loss'] = 'l1_corr'
    print("="*50)
    print("="*20, "testing", "="*20)

    from pprint import pprint
    pprint(config)
    print("="*50)

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(agent, env_spec, config)


  # -----------------------
  # define dict to save. add some extra stuff here
  # -----------------------
  save_config_dict = save_config_dict or dict()
  save_config_dict.update(config.__dict__)
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

  # observers = [LevelAvgReturnObserver()]
  observers = [LevelAvgReturnObserver(reset=return_avg_episodes)]
  # -----------------------
  # wandb settup
  # -----------------------
  os.chdir(path)
  setting = env_kwargs.get('setting', 'default')


  def get(label, default):
    return tasks_file.get(label, default)
  actor_label = get("actor_label", actor_label or f"actor_{setting}")
  evaluator_label = get("evaluator_label", evaluator_label or f"evaluator_{setting}")
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
    log_with_key='log_data',
    observers=observers,
    loss_label='Loss',
    actor_label=actor_label,
    evaluator_label=evaluator_label,
    **kwargs,
    )

def main(_):
  config_kwargs=dict(seed=FLAGS.seed)

  if FLAGS.max_number_of_steps is not None:
    config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps

  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    name=FLAGS.name if FLAGS.name else FLAGS.agent,
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
    update_wandb_name=False,
    env_kwargs=dict(
      setting=FLAGS.env_setting,
      task_reps=FLAGS.task_reps,
      symbolic=False,
    )
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
