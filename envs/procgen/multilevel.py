"""Class used for MultiLevel version of Kitchen Env.

Each level can have different distractors, different layout,
    different tasks, etc. Very flexible since just takes in 
    dict(level_name:level_kwargs).
"""
import numpy as np
import copy

from gym import spaces

import gym

class MultiLevel(object):

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
      level_names : dict,
      **kwargs):
    """Summary
    
    Args:
        level_names (dict): {levelname: kwargs} dictionary
        levelname2idx (dict, optional): {levelname: idx} dictionary. useful for returning idx versions of levelnames.
        LevelCls (TYPE, optional): Description
        wrappers (list, optional): Description
        **kwargs: kwargs for all levels

    """
    self.initialized = False
    self.kwargs = kwargs

    self.levels = dict()
    self.level_names = level_names
    self.names = list(level_names.keys())

    self.levelname2idx = levelname2idx or {k:idx for idx, k in enumerate(self.names)}

    self._current_idx = 0

  def get_level(self, idx):
    """Return level for idx. Spawn environment lazily.
    Args:
        idx (TYPE): idx to return (or spawn)
    
    Returns:
        TYPE: level
    """
    key = self.names[idx]
    if not key in self.levels:
      self.levels[key] = gym.make(self.level_names[key])

    return self.levels[key]

  def reset(self, **kwargs):
    """Sample new level."""
    self._current_idx = np.random.randint(len(self.names))
    obs = self.current_level.reset(**kwargs)

    return obs

  def step(self, *args, **kwargs):
    """Sample new level."""
    obs, reward, done, info = self.current_level.step(*args, **kwargs)
    return obs, reward, done, info


  @property
  def current_levelname(self):
      return self.names[self._current_idx]

  @property
  def current_level(self):
    return self.get_level(self._current_idx)
