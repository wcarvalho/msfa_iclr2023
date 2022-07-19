"""

"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from absl import app
from absl import flags
import acme

import functools

from agents import td_agent
from projects.common.train_distributed import build_common_program
from projects.skill_discovery import helpers
from projects.common.observers import LevelReturnObserver

from utils import gen_log_dir
# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_bool('evaluate', False, 'whether to use evaluation policy.')
flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')

# -----------------------
# wandb
# -----------------------
flags.DEFINE_bool('wandb', False, 'whether to log.')
flags.DEFINE_bool('init_only', False, 'whether to end after network initialization.')
flags.DEFINE_string('wandb_project', '', 'wand project.')
flags.DEFINE_string('wandb_entity', '', 'wandb username')
flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('notes', '', 'notes for wandb.')


FLAGS = flags.FLAGS


def build_program(
  agent: str,
  num_actors : int,
  wandb_init_kwargs=None,
  env_kwargs=None,
  hourminute=True, # whether to append hour-minute to logger path
  log_every=5.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  **kwargs,
  ):
  env_kwargs = env_kwargs or dict()
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment(
      evaluation=is_eval)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = helpers.load_agent_settings(agent, env_spec, config_kwargs)

  # -----------------------
  # define dict to save. add some extra stuff here
  # -----------------------
  save_config_dict = config.__dict__
  save_config_dict.update(
    agent=agent,
    **env_kwargs,
  )

  # -----------------------
  # data stuff:
  #   construct log directory if necessary
  #   + observer of data
  # -----------------------
  if not log_dir:
    log_dir, config_path_str = gen_log_dir(
      base_dir=f"{path}/results/skill_discovery/distributed",
      hourminute=hourminute,
      return_kwpath=True,
      seed=config.seed,
      agent=str(agent))


  observers = [LevelReturnObserver()]
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
    num_actors=num_actors,
    save_config_dict=save_config_dict,
    log_every=log_every,
    observers=observers,
    pregenerate_named_tuple=False,
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
    )

  # Launch experiment.
  controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    # below specifies to not use GPU for actor/evaluator
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
  )
  controller.wait()

if __name__ == '__main__':
  app.run(main)
