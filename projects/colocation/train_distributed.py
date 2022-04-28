"""
Run Successor Feature based agents and baselines on
  BabyAI derivative environments.

Command I run for r2d1:
  PYTHONPATH=$PYTHONPATH:$HOME/successor_features/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=3 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/colocation/train_distributed.py \
    --agent r2d1_noise --simple False --one_room False --nowalls False


Command for tensorboard
ssh -L 16006:127.0.0.1:6006 nameer@deeplearn9.eecs.umich.edu
source ~/.bashrc; conda activate acmejax; cd ~/successor_features/rljax/results/colocation/distributed/experiments
tensorboard --logdir .
"""
#most recent 2 are r2d1 and then r2d1_noise, both with no walls, 6 objects

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print(os.environ['LD_LIBRARY_PATH'])


import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.colocation import helpers
from projects.colocation.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir
import pickle
from projects.common.train_distributed import build_common_program
from projects.common.observers import LevelReturnObserver
from projects.colocation.observers import RoomReturnObserver, FullReturnObserver


# -----------------------
# flags
# -----------------------

flags.DEFINE_string('experiment', None, 'experiment_name.')
flags.DEFINE_bool('simple',False, 'should the environment be simple or have some colocation')
flags.DEFINE_bool('nowalls',False,'No doors in environment')
flags.DEFINE_bool('one_room',False, 'all in one room')
flags.DEFINE_string('agent', 'usfa', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')

flags.DEFINE_bool('wandb', False, 'whether to log.')
flags.DEFINE_string('wandb_project', 'msf2', 'wand project.')
flags.DEFINE_string('wandb_entity', 'wcarvalho92', 'wandb entity')
flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('wandb_notes', '', 'notes for wandb.')

FLAGS = flags.FLAGS

def build_program(
  agent: str,
  num_actors : int,
  wandb_init_kwargs=None,
  update_wandb_name=True, # use path from logdir to populate wandb name
  setting='small',
  group='experiments', # subdirectory that specifies experiment group
  hourminute=True, # whether to append hour-minute to logger path
  log_every=30.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  is_simple: bool = True,
  nowalls: bool = False,
  one_room: bool = False,
    ):
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment_sanity_check(evaluation=is_eval, simple=is_simple, agent=agent, nowalls=nowalls, one_room=one_room)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings_sanity_check(env_spec, agent=FLAGS.agent)

  #observers
  observers = [LevelReturnObserver(), RoomReturnObserver(),FullReturnObserver()]


  save_config_dict = config.__dict__
  save_config_dict.update(
    agent=agent,
    setting=setting,
    group=group
  )

  if not log_dir:
    log_dir, config_path_str = gen_log_dir(
      base_dir=f"{path}/results/colocation/distributed/{group}",
      hourminute=hourminute,
      return_kwpath=True,
      seed=config.seed,
      agent=str(agent))

    if wandb_init_kwargs and update_wandb_name:
      wandb_init_kwargs['name'] = config_path_str

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
    loss_label=loss_label,
    num_actors=num_actors,
    save_config_dict=save_config_dict,
    log_every=log_every,
    observers=observers
    #envloop_class=EnvironmentLoop,
    )

def main(_):

  config_kwargs = dict(seed=FLAGS.seed)

  if FLAGS.max_number_of_steps is not None:
      config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      group=FLAGS.group if FLAGS.group else FLAGS.agent,  # organize individual runs into larger experiment
      notes=FLAGS.wandb_notes,
  )

  program = build_program(
      agent=FLAGS.agent,
      num_actors=FLAGS.num_actors,
      config_kwargs=config_kwargs,
      wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
      is_simple=FLAGS.simple,
      nowalls=FLAGS.nowalls,
      one_room=FLAGS.one_room
  )

  # Launch experiment.
  controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
                         terminal='current_terminal',
                         local_resources={
                             'actor':
                                 PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
                             'evaluator':
                                 PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
                         )
  controller.wait()

if __name__ == '__main__':
  app.run(main)
