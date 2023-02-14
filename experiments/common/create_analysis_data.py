"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/msfa/lib/ \
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
from typing import Callable, Optional, Sequence, Tuple, Union

import json
from glob import glob
import os.path
from absl import app
from absl import flags
import acme
import functools
import jax
import dm_env

from acme.agents import agent as acme_agent
from acme.jax import savers
from agents import td_agent
from experiments.common.train import run
from utils import make_logger, gen_log_dir
from acme.utils import paths
from acme.environment_loop import EnvironmentLoop
from acme.jax.layouts import distributed_layout
import numpy as np

import matplotlib.animation as animation
from experiments.common.train import create_net_prediction_tuple
from experiments.common.loading import load_agent, load_ckptr
from experiments.exploration2 import helpers
from experiments.exploration2 import video_utils


# -----------------------
# flags
# -----------------------
# flags.DEFINE_string('agent', 'r2d1', 'which agent.')
# flags.DEFINE_string('env_setting', 'large_respawn', 'which environment setting.')
# flags.DEFINE_integer('num_episodes', int(5), 'Number of episodes to evaluate agents for.')
flags.DEFINE_integer('ckpts', -1, '-1: latest. 0: all. {interger}: that corresponding checkpoint')
flags.DEFINE_string('video_path', None, '')
flags.DEFINE_bool('overwrite', True, 'whether to use evaluation policy.')


FLAGS = flags.FLAGS


from acme.wrappers.video import VideoWrapper, make_animation

class OARVideoWrapper(VideoWrapper):
  """docstring for OARVideoWrapper"""

  def __init__(self,
               environment: dm_env.Environment,
               *,
               path: str = '~/acme',
               filename: str = '',
               process_path: Callable[[str, str], str] = paths.process_path,
               record_every: int = 100,
               frame_rate: int = 30,
               dpi=150,
               figsize: Optional[Union[float, Tuple[int, int]]] = None):
    super(VideoWrapper, self).__init__(environment)
    self._path = process_path(path, 'videos', add_uid=False)
    self._filename = filename
    self._record_every = record_every
    self._frame_rate = frame_rate
    self._frames = []
    self._counter = 0
    self._figsize = figsize
    self.level = None
    self.dpi = dpi

  def _render_frame(self, observation):
    """Renders a frame from the given environment observation."""
    return observation.observation.image


  def _write_frames(self):
    """Writes frames to video."""
    if self._counter % self._record_every == 0:
      self.level = str(self.environment.env.current_levelname)
      path = os.path.join(self._path,
                          f'{self._filename}_{self.level}_{self._counter:04d}.mp4')

      ani = make_animation(self._frames, self._frame_rate,
                             self._figsize)

      # with open(path, 'w') as f:
      #   f.write(video)
      Writer = animation.writers['ffmpeg']
      writer = Writer(fps=self._frame_rate, metadata=dict(artist='Me'), bitrate=1800)
      ani.save(path, writer, dpi=self.dpi)
      # Clear the frame buffer whether a video was generated or not.
      self._frames = []


def first_path(path_search):
  options = glob(path_search)
  print("Selected", options[0])
  return options[0]


def load_agent_ckpt(
  settings : dict,
  agent_name : str,
  ckpt_dir,
  results_path=None,
  overwrite=True,
  video_observer=True,
  predict_cumulants=True,
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

  config = settings['config']
  NetworkCls = settings['NetworkCls']
  NetKwargs = settings['NetKwargs']
  LossFn = settings['LossFn']
  LossFnKwargs = settings['LossFnKwargs']
  env_spec = settings['env_spec']

  # -----------------------
  # data utils
  # -----------------------
  observer = None 
  if video_observer:
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
      predict_cumulants=predict_cumulants,
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


def collect_data(env, agent: acme_agent, observers, total_episodes: int):
  # -----------------------
  # make env + run
  # -----------------------
  loop = EnvironmentLoop(
    env,
    agent,
    should_update=False,
    observers=observers
    )
  loop.run(total_episodes+1)


def generate_data(agent_name, agent_data_path, load_env_agent_settings, basepath='.', overwrite=True, ckpt=-1, total_episodes=5, seed=None, frame_rate=10, make_videos_every=1, video_path=None):
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

  # for seed_path in seed_paths:
  seed_path = seed_paths[0]
  config = first_path(os.path.join(seed_path, "config.json"))
  with open(config, 'r') as f:
    config = json.load(f)


  # -----------------------
  # load env
  # -----------------------
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

  def make_video_path(seed_path):
    short_seed_path = seed_path.split(basepath)[1]
    if video_path:
      path = os.path.join(video_path, short_seed_path)
    else:
      path = os.path.join(basepath,'videos', short_seed_path)

    return path

  if ckpt == -1:
    # latest
    results_path = os.path.join(seed_path, 'episode_data/latest')
    video_observer.set_results_path(results_path)

    if make_videos_every:
      env = OARVideoWrapper(
        environment=env,
        frame_rate=frame_rate,
        path=make_video_path(data_path),
        record_every=make_videos_every,
        filename='vid')

    collect_data(
      env=env,
      agent=agent,
      observers=[video_observer],
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
        observers=[video_observer],
        total_episodes=total_episodes)

  else:
    load_ckpt(ckpt)
    collect_data(
      env=env,
      agent=agent,
      observers=[video_observer],
      total_episodes=total_episodes)
