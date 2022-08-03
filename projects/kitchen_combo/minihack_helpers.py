import acme
import functools

from acme import wrappers
import cv2
import dm_env
import gym
import rlax

from typing import NamedTuple

from acme import types
from utils import ObservationRemapWrapper
from utils import data as data_utils
from gym import spaces
import numpy as np
import minihack

from agents import td_agent
from agents.td_agent import losses

from losses import usfa as usfa_losses
from losses.vae import VaeAuxLoss
from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss
from losses import cumulants
from modules.ensembles import QLearningEnsembleLoss

from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper


# from projects.msf import nets
from projects.kitchen_gridworld import helpers as kitchen_helpers
from projects.msf import helpers as msf_helpers
from projects.kitchen_combo import minihack_configs
from projects.common_usfm import agent_loading
# from projects.common_usfm import nets

from envs.gym_multitask import GymTask, MultitaskGym, MultiLevelEnv

def downsample(x, shape, interpolation=cv2.INTER_LINEAR):
  return cv2.resize(x, dsize=shape, interpolation=interpolation)

class MinihackGymTask(GymTask):
  """docstring for GymTask"""
  def __init__(self, env, task, num_seeds=0, image_shape=(64,64)):
    super(GymTask, self).__init__()
    from minihack import RewardManager

    reward_manager = RewardManager()
    self._env = gym.make(env, observation_keys=("pixel_crop",), penalty_step=0.0)

    self.task = np.array(task, dtype=np.float32)
    self.num_seeds = num_seeds
    self.image_shape = image_shape

    # custom observation space
    image_space = spaces.Box(
        low=0,
        high=255,
        shape=image_shape + (3,),
        dtype=np.uint8,
    )
    task_space = spaces.Box(
        low=0,
        high=255,
        shape=self.task.shape,
        dtype=self.task.dtype
    )
    self.observation_space = spaces.Dict({
        'image': image_space,
        'task': task_space,
    })


  def __getattr__(self, name):
    """This is where all the magic happens. 
    This enables this class to act like `env`."""
    return getattr(self._env, name)

  def step(self, action):
    obs_dict, reward, done, info = self._env.step(int(action))
    obs=dict(
      image=downsample(obs_dict['pixel_crop'], self.image_shape),
      task=self.task)
    return obs, reward, done, info

  def reset(self):
    if self.num_seeds:
      seed = np.random.randint(self.num_seeds)
      self._env.seed(seed)
      obs_dict = self._env.reset()
    else:
      obs_dict = self._env.reset()
    obs=dict(
      image=downsample(obs_dict['pixel_crop'], self.image_shape),
      task=self.task)
    return obs

def make_environment(
  setting='',
  evaluation: bool = False,
  num_train_seeds=200,
  **kwargs) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  setting = setting or 'room_small'
  assert setting in [
    'room_small',
    'room_large',
    ]

  # -----------------------
  # environments
  # -----------------------
  if setting in ['room_small', 'room_large']:
    size='15x15' if setting == 'room_large' else '5x5'
    train_level_kwargs={
        'a.train|1,0,0|': dict(
          env=f'MiniHack-Room-Monster-{size}-v0', task=[1,0,0]),
        'a.train|0,1,0|': dict(
          env=f'MiniHack-Room-Trap-{size}-v0', task=[0,1,0]),
        'a.train|0,0,1|': dict(
          env=f'MiniHack-Room-Dark-{size}-v0', task=[0,0,1]),
      }
    if evaluation:
      all_level_kwargs={
        'b.eval|1,1,1|': dict(
          env=f'MiniHack-Room-Ultimate-{size}-v0', task=[1,1,1]),
        **train_level_kwargs
      }
      num_seeds=0
    else:
      all_level_kwargs=train_level_kwargs
      num_seeds = num_train_seeds

  env = MultitaskGym(
    all_level_kwargs=all_level_kwargs,
    EnvCls=MinihackGymTask,
    # distribution_mode=setting,
    num_seeds=num_seeds,
    )

  wrapper_list = [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)


def load_agent_settings(agent, env_spec, config_kwargs=None, env_kwargs=None):
  default_config = dict()
  default_config.update(config_kwargs or {})

  agent = agent.lower()
  if agent == "r2d1":
  # Recurrent DQN/UVFA
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[minihack_configs.R2D1Config()],
      )

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning

    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.usfa_lstm(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          minihack_configs.QAuxConfig(),
          minihack_configs.RewardConfig(),
          minihack_configs.USFAConfig(),
          ],
      )


  elif agent == "msf":
  # USFA + cumulants from FARM + Q-learning (1.9M)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.msf(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          minihack_configs.QAuxConfig(),
          minihack_configs.RewardConfig(),
          minihack_configs.ModularUSFAConfig(),
          minihack_configs.FarmConfig(),
        ],
      )

  else:
    raise NotImplementedError(agent)

  loss_label=None
  eval_network=False
  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
