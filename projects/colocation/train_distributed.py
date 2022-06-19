"""
Run Successor Feature based agents and baselines on
  BabyAI derivative environments.

Command I run to train:
  PYTHONPATH=$PYTHONPATH:$HOME/successor_features/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    WANDB_START_METHOD="thread" \
    python projects/colocation/train_distributed.py \
    --agent msf --room_reward .25 --wandb_name 6-13 --seed 1


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
from projects.colocation.observers import RoomReturnObserver, FullReturnObserver, PickupCountObserver
import string
import random


# -----------------------
# flags
# -----------------------

flags.DEFINE_string('wandb_name', None, 'wandb name')
flags.DEFINE_bool('simple',False, 'should the environment be simple or have some colocation')
flags.DEFINE_bool('nowalls',False,'No doors in environment')
flags.DEFINE_bool('one_room',False, 'all in one room')
flags.DEFINE_bool('deterministic_rooms',False,'rooms are not in random order')
flags.DEFINE_float('room_reward',0,'reward for entering the correct room')
flags.DEFINE_integer('train_task_as_z', 0, '0 for None, -1 for no, 1 for yes')


flags.DEFINE_string('agent', 'msf', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')

flags.DEFINE_bool('wandb', True, 'whether to log.')
flags.DEFINE_string('wandb_project', 'successor_features', 'wand project.')
flags.DEFINE_string('wandb_entity', 'nrocketmann', 'wandb entity')
flags.DEFINE_string('wandb_notes', '', 'notes for wandb.')

FLAGS = flags.FLAGS

def build_program(
  agent: str,
  num_actors : int,
  wandb_init_kwargs=None,
  wandb_name = None,
  setting='small',
  group='experiments', # subdirectory that specifies experiment group
  hourminute=True, # whether to append hour-minute to logger path
  log_every=30.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  simple: bool = True,
  nowalls: bool = False,
  one_room: bool = False,
  deterministic_rooms: bool = False,
  room_reward: float = 0,
  train_task_as_z: int = 0,
  randomize_name: bool = True
    ):
  if train_task_as_z==0:
      train_task_as_z = None
  elif train_task_as_z==-1:
      train_task_as_z = False
  elif train_task_as_z==1:
      train_task_as_z=True
  else:
      raise NotImplementedError("Invalid value for train_task_as_z: {0}".format(train_task_as_z))

  room_reward_task_vector = True #in oracle case we want task vector to have room reward
  if agent in ['usfa_conv','usfa_lstm','msf','conv_msf']:
      room_reward_task_vector = False
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment_sanity_check(evaluation=is_eval, simple=simple, agent=agent,
                                      nowalls=nowalls, one_room=one_room, deterministic_rooms=deterministic_rooms,
                                              room_reward=room_reward, room_reward_task_vector=room_reward_task_vector)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings_sanity_check(env_spec, agent=agent, train_task_as_z=train_task_as_z, config_kwargs=config_kwargs)

  #observers
  observers = [LevelReturnObserver(), RoomReturnObserver(),FullReturnObserver(), PickupCountObserver()]


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

  # generate unique hash for name

    if wandb_name is None:
        wandb_name = 'test'
    wandb_name+= '-' + agent
    if train_task_as_z or (train_task_as_z is None and (agent=='usfa_lstm' or agent=='usfa_conv')):
        wandb_name+='-new_sampling'
    else:
        wandb_name+='-og_sampling'
    if simple:
        wandb_name+='-simple'
    if one_room:
        wandb_name+='-one_room'
    if nowalls:
        wandb_name+='-no_walls'
    if deterministic_rooms:
        wandb_name+='-deterministic_rooms'
    if room_reward!=0:
        wandb_name+='-room_reward'
    if room_reward_task_vector:
        wandb_name+='-oracle'
    if randomize_name:
        letters = string.ascii_lowercase
        hashcode = ''.join(random.choice(letters) for _ in range(10))
        wandb_name+=hashcode


    if wandb_init_kwargs:
      wandb_init_kwargs['name'] = wandb_name
      wandb_init_kwargs['config'] = dict(
      is_simple=simple,
      nowalls=nowalls,
      one_room=one_room,
      deterministic_rooms=deterministic_rooms,
      room_reward=room_reward
      )

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
    )

def main(_):

  config_kwargs = dict(seed=FLAGS.seed)

  if FLAGS.max_number_of_steps is not None:
      config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      group=FLAGS.agent,  # organize individual runs into larger experiment
      notes=FLAGS.wandb_notes
  )

  program = build_program(
      agent=FLAGS.agent,
      num_actors=FLAGS.num_actors,
      config_kwargs=config_kwargs,
      wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
      simple=FLAGS.simple,
      nowalls=FLAGS.nowalls,
      one_room=FLAGS.one_room,
      deterministic_rooms=FLAGS.deterministic_rooms,
      room_reward=FLAGS.room_reward,
      wandb_name=FLAGS.wandb_name,
      train_task_as_z=FLAGS.train_task_as_z
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
