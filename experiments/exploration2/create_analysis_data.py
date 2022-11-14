"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python -m ipdb -c continue experiments/exploration1/goto.py \
    --agent r2d1_noise

"""

# Do not preallocate GPU memory for JAX.

import launchpad as lp
import logging
from launchpad.nodes.python.local_multi_processing import PythonProcess

import json
from glob import glob
import os.path
from absl import app
from absl import flags
import acme
import functools
import jax

from acme.agents import agent as acme_agent
from acme.jax import savers
from agents import td_agent
from experiments.common.train import run
from utils import make_logger, gen_log_dir
from acme.environment_loop import EnvironmentLoop
from acme.jax.layouts import distributed_layout
import numpy as np

from experiments.common.train import create_net_prediction_tuple
from experiments.common.loading import load_agent, load_ckptr
from experiments.exploration2 import helpers
from experiments.exploration2 import video_utils


# -----------------------
# flags
# -----------------------
# flags.DEFINE_string('agent', 'r2d1', 'which agent.')
# flags.DEFINE_string('env_setting', 'large_respawn', 'which environment setting.')
flags.DEFINE_integer('num_episodes', int(5), 'Number of episodes to evaluate agents for.')
flags.DEFINE_integer('ckpts', -1, '-1: latest. 0: all. {interger}: that corresponding checkpoint')
flags.DEFINE_bool('overwrite', True, 'whether to use evaluation policy.')


FLAGS = flags.FLAGS




def first_path(path_search):
  options = glob(path_search)
  print("Selected", options[0])
  return options[0]


def load_kitchen_agent(
  config_kwargs : dict,
  agent_name : str,
  env_spec,
  max_vocab_size,
  ckpt_dir,
  results_path=None,
  overwrite=True,
  observer_episodes=1):
  """
  Changes:
      1. custom observer
      2. custom behavior policy
      3. wrapper to actor
  
  Args:
      config_kwargs (TYPE): Description
      agent_name (TYPE): Description
      env_spec (TYPE): Description
      max_vocab_size (TYPE): Description
      results_path (TYPE): Description
      ckpt_dir (TYPE): Description
      overwrite (bool, optional): Description
      observer_episodes (int, optional): Description
  
  Returns:
      TYPE: Description
  """

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(
    agent=config_kwargs['agent'],
    env_spec=env_spec,
    max_vocab_size=max_vocab_size,
    config_kwargs=config_kwargs)

  # -----------------------
  # data utils
  # -----------------------
  observer = video_utils.DataStorer(
    results_path=results_path,
    agent=agent_name,
    seed=config.seed,
    episodes=observer_episodes,
    exit=True)

  behavior_policy_constructor = functools.partial(video_utils.make_behavior_policy, evaluation=True)

  def wrap_actor(actor, networks, learner):
    return video_utils.ActorStorageWrapper(
      agent=actor,
      networks=networks,
      observer=observer,
      epsilon=config.evaluation_epsilon,
      seed=config.seed)

  # -----------------------
  # load agent
  # -----------------------
  agent, checkpointer = load_agent(
    config=config,
    env_spec=env_spec,
    NetworkCls=NetworkCls,
    NetKwargs=NetKwargs,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    directory=ckpt_dir,
    behavior_policy_constructor=behavior_policy_constructor,
    wrap_actor=wrap_actor,
    )

  return agent, observer, checkpointer


def collect_data(env, agent: acme_agent, observer, total_episodes: int):
  # -----------------------
  # make env + run
  # -----------------------
  loop = EnvironmentLoop(
    env,
    agent,
    should_update=False,
    observers=[observer]
    )
  loop.run(total_episodes+1)


def main(_):
  basepath = './results/kitchen_grid/final'
  searches = dict(
      msf='conv_msf3/agen=exploration1,sett=multiv9,cumu=lstm,memo=512,modu=None,modu=1,nmod=8', #
    )

  for agent, agent_search in searches.items():
    # -----------------------
    # get path info and load config
    # -----------------------
    path_search = os.path.join(basepath, agent_search)
    path = first_path(path_search)

    seed_paths = glob(os.path.join(path, "seed=*"))
    seed_paths.sort()

    cuda_idx = 0
    # for seed_path in seed_paths:
    seed_path = seed_paths[2]
    config = first_path(os.path.join(seed_path, "config.json"))
    with open(config, 'r') as f:
      config = json.load(f)


    # -----------------------
    # load env
    # -----------------------
    env = helpers.make_environment(
      setting=config['setting'],
      task_reps=config.get('task_reps', 'pickup'),
      evaluation=False # test set (harder)
      )
    max_vocab_size = len(env.env.instr_preproc.vocab) # HACK
    nlevels = len(env.env.all_level_kwargs)
    total_episodes=nlevels*FLAGS.num_episodes
    env_spec = acme.make_environment_spec(env)

    # -----------------------
    # load agent
    # -----------------------
    dirs = glob(os.path.join(seed_path, "2022*")); dirs.sort()
    agent, observer, checkpointer = load_kitchen_agent(
      config_kwargs=config,
        agent_name=agent,
        env_spec=env_spec,
        max_vocab_size=max_vocab_size,
        ckpt_dir=dirs[-1],
        observer_episodes=total_episodes,
        overwrite=FLAGS.overwrite)

    ckpts = checkpointer._checkpointer._checkpoint_manager.checkpoints
    ckpt_dir = checkpointer._checkpointer._checkpoint_manager.directory

    def load_ckpt(ckpt):
      ckpt_path = f'{ckpt_dir}/ckpt-{ckpt}'
      assert os.path.exists(f'{ckpt_path}.index')
      logging.info('Attempting to restore checkpoint: %s',
                 ckpt_path)
      checkpointer._checkpointer._checkpoint.restore(ckpt_path)

      observer.set_results_path(os.path.join(seed_path, f'episode_data/ckpt-{ckpt}'))

    ckpt = FLAGS.ckpts
    if ckpt == -1:
      # latest
      observer.set_results_path(os.path.join(seed_path, 'episode_data/latest'))
      collect_data(
        env=env,
        agent=agent,
        observer=observer,
        total_episodes=total_episodes)

    elif ckpt == 0:
      # all
      # make dictionary of idx --> ckpt
      # sort from start to end
      # loop from start to end
      import ipdb; ipdb.set_trace()
      def make_key(c):
        end = c.split("/")[-1]
        idx = end.split("ckpt-")[-1]
        return idx
      ckpts = {make_key(c):c for c in ckpts}
      idxs = sort(list(ckpts.keys()))
      for idx in idxs:
        load_ckpt(idx)
        collect_data(
          env=env,
          agent=agent,
          observer=observer,
          total_episodes=total_episodes)

    else:
      load_ckpt(ckpt)
      collect_data(
        env=env,
        agent=agent,
        observer=observer,
        total_episodes=total_episodes)

    #   generate_latest(
    #     config_kwargs=config,
    #     agent_name=agent,
    #     results_path=os.path.join(seed_path, 'episode_data/latest'),
    #     ckpt_dir=ckpt_dir,
    #     num_episodes=FLAGS.num_episodes,
    #     overwrite=FLAGS.overwrite)
    # elif FLAGS.ckpts == 'all':
    #   import ipdb; ipdb.set_trace()



if __name__ == '__main__':
  app.run(main)
