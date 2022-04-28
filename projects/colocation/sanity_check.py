

"""HIGH LEVEL GOAL: Train an agent to do things in the colocated environment at a very basic level.
TASK1: With no colocation (1 object per room), doors open by default, train an R2D1 agent to GOTO each object
TASK1: With some colocation (2-3 objects per room), doors open by default, train an agent to GOTO each object"""

#SETUP THINGS
#conda activate acmejax
#conda develop /path/to/rljax
#export LD_LIBRARY_PATH=/home/nameer/miniconda3/envs/acmejax/lib
#cd successor_features/rljax
#python projects/colocation/sanity_check.py

"""Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/successor_features/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/colocation/sanity_check.py"""

import os
#os.environ['LD_LIBRARY_PATH'] = "/home/nameer/miniconda3/envs/acmejax/lib"
#os.environ["PYTHONPATH"]="${PYTHONPATH}:/home/nameer/successor_features/rljax"
#os.environ['CUDA_VISIBLE_DEVICES']="0"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.colocation import helpers
from projects.colocation.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent','r2d1_noise','what kind of agent? r2d1 or usfa available now')
flags.DEFINE_bool('super_simple',False,'1 object per room, or a bit of colocation?')
flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('evaluate', False, 'whether to use evaluation policy.')

FLAGS = flags.FLAGS

def main(_):
  env = helpers.make_environment_sanity_check(evaluation=False, simple=FLAGS.super_simple,agent=FLAGS.agent,nowalls=False, one_room=False) #3 objects if super simple, otherwise 2-3 types
  env_spec = acme.make_environment_spec(env)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs,_,_ = helpers.load_agent_settings_sanity_check(env_spec,agent=FLAGS.agent)

  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/colocation_sanity_check/local",
    agent=FLAGS.agent,
    seed=config.seed)

  logger_fn = lambda : make_logger(
    log_dir=log_dir, label=FLAGS.agent)


  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
      logger_fn=logger_fn)

  kwargs={}
  if FLAGS.evaluate:
    kwargs['behavior_policy_constructor'] = functools.partial(td_agent.make_behavior_policy, evaluation=True)
  agent = td_agent.TDAgent(
      env_spec,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True),
      builder=builder,
      workdir=log_dir,
      config=config,
      seed=FLAGS.seed,
      **kwargs,
      )

  # -----------------------
  # make env + run
  # -----------------------
  env_logger = make_logger(
    log_dir=log_dir,
    label='actor',
    steps_key="steps")

  loop = EnvironmentLoop(env, agent, logger=env_logger)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
