from acme import wrappers
import cv2
import dm_env
import gym

from gym import spaces
import numpy as np

# from experiments.exploration1 import nets
from experiments.iclr2023 import minihack_configs
from experiments.common import agent_loading

from envs.gym_multitask import GymTask, MultitaskGym


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
        # **train_level_kwargs
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

  return agent_loading.default_agent_settings(agent=agent,
                                              env_spec=env_spec,
                                              configs=minihack_configs,
                                              config_kwargs=config_kwargs,
                                              env_kwargs=env_kwargs)

