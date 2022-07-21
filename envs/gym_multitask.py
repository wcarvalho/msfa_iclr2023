"""

Each level can have different distractors, different layout,
    different tasks, etc. Very flexible since just takes in 
    dict(level_name:level_kwargs).
"""

from typing import NamedTuple, Sequence

import dm_env
from dm_env import specs

from acme import types
from acme.wrappers import GymWrapper

import numpy as np

from gym import spaces
import gym


class GymObsTuple(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  task: types.Nest
  train_tasks: types.Nest

class GymTask(object):
  """docstring for GymTask"""
  def __init__(self, env, task):
    super(GymTask, self).__init__()
    self._env = gym.make(env)

    self.task = np.array(task, dtype=np.int32)
    # custom observation space
    image_space = self._env.observation_space
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
    image, reward, done, info = self._env.step(int(action))
    obs=dict(
      image=image,
      task=self.task)
    return obs, reward, done, info

  def reset(self):
    image = self._env.reset()
    obs=dict(
      image=image,
      task=self.task)
    return obs


class MultiLevelEnv(object):

  """Wrapper environment that acts like the `current_level`.
  Everytime reset is called, a new level is sampled.
  Attributes:
      levelnames (list): names of levels
      levels (dict): level objects
  """

  def __getattr__(self, name):
    """This is where all the magic happens. 
    This enables this class to act like `current_level`."""
    return getattr(self.current_level, name)

  def __init__(self,
      all_level_kwargs : dict,
      EnvCls : GymTask=GymTask,
      **kwargs):
    """Summary
    
    Args:
        all_level_kwargs (dict): {levelname: kwargs} dictionary
        levelname2idx (dict, optional): {levelname: idx} dictionary. useful for returning idx versions of levelnames.
        LevelCls (TYPE, optional): Description
        wrappers (list, optional): Description
        **kwargs: kwargs for all levels

    """
    self.initialized = False
    self.kwargs = kwargs
    self.EnvCls = EnvCls
    self._current_idx = 0

    self.levels = dict()
    self.all_level_kwargs = all_level_kwargs
    self.levelnames = list(all_level_kwargs.keys())

    self.levelname2idx = {k:idx for idx, k in enumerate(self.levelnames)}

    train_tasks = [self.get_level(idx).task for idx in range(len(self.levelnames))]
    self.train_tasks = np.array(train_tasks,
      dtype=self.observation_space['task'].dtype)

    self.observation_space = spaces.Dict({
        'image': self.observation_space['image'],
        'task': self.observation_space['task'],
        'train_tasks': spaces.Box(
            low=0, high=255,
            shape=self.train_tasks.shape,
            dtype=self.train_tasks.dtype)
    })


  def get_level(self, idx):
    """Return level for idx. Spawn environment lazily.
    Args:
        idx (TYPE): idx to return (or spawn)
    
    Returns:
        TYPE: level
    """
    key = self.levelnames[idx]
    if not key in self.levels:
      kwargs = self.all_level_kwargs[key]
      kwargs.update(self.kwargs)
      # env = gym.make(kwargs['name'])
      env = self.EnvCls(**kwargs)
      self.levels[key] = env

    return self.levels[key]

  def reset(self, **kwargs):
    """Sample new level."""
    self._current_idx = np.random.randint(len(self.levelnames))
    obs = self.current_level.reset(**kwargs)
    obs['train_tasks'] = self.train_tasks

    return obs

  def step(self, *args, **kwargs):
    """Sample new level."""
    obs, reward, done, info = self.current_level.step(*args, **kwargs)
    obs['train_tasks'] = self.train_tasks

    return obs, reward, done, info


  @property
  def current_levelname(self):
      return self.levelnames[self._current_idx]

  @property
  def current_level(self):
    return self.get_level(self._current_idx)


class MultitaskGym(dm_env.Environment):
  """
  """

  def __init__(self,
    all_level_kwargs: dict, 
    ObsTuple:GymObsTuple=GymObsTuple,
    MultilevelCls: MultiLevelEnv=MultiLevelEnv,
    obs_keys=None,
    **kwargs):
    """Initializes a Multitask environment environment.
    
    Args:
        all_level_kwargs (dict): initialization arguments for even task environment
        ObsTuple (GymObsTuple, optional): Tuple used to create observations
        LevelCls (TYPE, optional): class used for each level
        room_size (int, optional): Description
        agent_view_size (int, optional): Description
        path (str, optional): root path from where script is run
        tile_size (int, optional): Description
        obs_keys (list, optional): keys for ObsTuple
        wrappers (None, optional): environment wrappers
        **kwargs: Description
    """

    self.env = MultilevelCls(
      all_level_kwargs=all_level_kwargs,
      **kwargs)

    self.default_env = GymWrapper(self.env)

    self.obs_keys = obs_keys or ObsTuple._fields
    self.ObsTuple = ObsTuple


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    obs = self.env.reset()
    obs = self.ObsTuple(**{k: obs[k] for k in self.obs_keys})
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    obs = self.ObsTuple(**{k: obs[k] for k in self.obs_keys})

    if done:
      timestep = dm_env.termination(reward=reward, observation=obs)
    else:
      timestep = dm_env.transition(reward=reward, observation=obs)

    return timestep


  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return self.default_env.action_spec()

  def observation_spec(self):
    default = self.default_env.observation_spec()
    return self.ObsTuple(**default)
