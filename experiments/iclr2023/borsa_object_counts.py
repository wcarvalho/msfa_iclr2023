"""
"""
import os
import logging
from absl import app
from absl import flags
import acme
from glob import glob
import os.path
import json
from absl import flags

FLAGS = flags.FLAGS

from experiments.common.create_analysis_data import OARVideoWrapper, first_path, load_agent_ckpt, collect_data
from experiments.common.observers import EvalCountObserver

from experiments.iclr2023 import borsa_helpers

def load_env_agent_settings(config : dict, setting=None, evaluate=True):
  """This can be shared across seeds"""
  setting = setting or config['setting']

  env = borsa_helpers.make_environment(
    setting=setting,
    evaluation=evaluate)

  env_spec = acme.make_environment_spec(env)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = borsa_helpers.load_agent_settings(
    config['agent'], env_spec, config_kwargs=config)

  settings=dict(
      config=config,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      LossFn=LossFn,
      LossFnKwargs=LossFnKwargs,
      env_spec=env_spec,
    )

  return env, settings

def generate_data(agent_name, agent_data_path, load_env_agent_settings, basepath='.', overwrite=True, ckpt=-1, total_episodes=1000, seed=None, make_videos_every=0, predict_cumulants=True, video_path=None):
  """Steps:
    1. load_env_agent_settings
    2. load_agent_ckpt (creates observer for making videos & storing stats)
    3. collect_data

  Args:
      agent_name (TYPE): Description
      agent_data_path (TYPE): Description
      load_env_agent_settings (TYPE): Description
      basepath (str, optional): Description
      overwrite (bool, optional): Description
      ckpt (TYPE, optional): Description
      total_episodes (int, optional): Description
      seed (None, optional): Description
      make_videos_every (int, optional): Description
      video_path (None, optional): Description
  """
  data_path = os.path.join(basepath, agent_data_path)
  # -----------------------
  # get path info and load config
  # -----------------------
  path = first_path(data_path)

  if seed:
    seed_paths = glob(os.path.join(path, f"seed={seed}"))
  else:
    seed_paths = glob(os.path.join(path, "seed=*"))
    seed_paths.sort()

  # # for seed_path in seed_paths:
  # seed_path = seed_paths[0]



  for seed_path in seed_paths:
    # -----------------------
    # load env + settings
    # -----------------------
    config = first_path(os.path.join(seed_path, "config.json"))
    with open(config, 'r') as f:
      config = json.load(f)
    env, settings = load_env_agent_settings(config=config)

    # -----------------------
    # load agent
    # -----------------------
    dirs = glob(os.path.join(seed_path, "2022*")); dirs.sort()
    agent, video_observer, checkpointer = load_agent_ckpt(
      settings=settings,
      agent_name=agent_name,
      ckpt_dir=dirs[-1],
      observer_episodes=total_episodes,
      predict_cumulants=predict_cumulants,
      video_observer=make_videos_every>0,
      overwrite=overwrite)

    ckpts = checkpointer._checkpointer._checkpoint_manager.checkpoints
    ckpt_dir = checkpointer._checkpointer._checkpoint_manager.directory

    def load_ckpt(ckpt):
      ckpt_path = f'{ckpt_dir}/ckpt-{ckpt}'
      assert os.path.exists(f'{ckpt_path}.index')
      logging.info('Attempting to restore checkpoint: %s',
                 ckpt_path)
      checkpointer._checkpointer._checkpoint.restore(ckpt_path)

      observer.set_results_path(os.path.join(seed_path, f'episode_data/ckpt-{ckpt}'))

    # -----------------------
    # video stuff
    # -----------------------
    def make_video_path(seed_path):
      short_seed_path = seed_path.split(basepath)[1]
      if video_path:
        return os.path.join(video_path, short_seed_path)
      else:
        return os.path.join(basepath,'videos', short_seed_path)


    # latest
    results_path = os.path.join(seed_path, 'episode_data/latest')
    if video_observer:
      video_observer.set_results_path(results_path)

    if make_videos_every and video_observer:
      env = OARVideoWrapper(
        environment=env,
        path=make_video_path(data_path),
        record_every=make_videos_every,
        filename='vid')

    # -----------------------
    # counts
    # -----------------------
    counts_observer = EvalCountObserver(
            path=seed_path,
            exit=False,
            agent=agent_name,
            seed=config['seed'],
            reset=total_episodes-1)

    collect_data(
      env=env,
      agent=agent,
      observers=[counts_observer],
      total_episodes=total_episodes)

def main(_):
  basepath = './results/msf2/'
  searches = dict(
      # exploration1='ablate_shared-4/agen=exploration1,memo=None,modu=150,sepe=False,sepe=False',
      # usfa='xl_respawn-grad-3/agen=usfa',
      # uvfa='xl_respawn-grad-3/agen=r2d1',
      # usfa_lstm='xl_respawn-grad-3/agen=usfa_lstm',
      uvfa_farm='xl_respawn-farm-4/agen=r2d1_farm*True*False',
    )

  for agent, agent_search in searches.items():
    generate_data(
      agent_name=agent,
      basepath=basepath,
      agent_data_path=agent_search,
      predict_cumulants=agent in ['usfa_lstm', 'exploration1'],
      load_env_agent_settings=load_env_agent_settings,
      overwrite=FLAGS.overwrite,
      ckpt=FLAGS.ckpts,
      video_path=FLAGS.video_path)


if __name__ == '__main__':
  app.run(main)
