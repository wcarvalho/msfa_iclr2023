import abc
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Sequence, Union

from acme.utils.loggers.base import Logger
from acme.utils.observers import EnvLoopObserver

import dm_env
from dm_env import specs
import jax.numpy as jnp
import numpy as np
import operator
import tree

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

Number = Union[int, float, np.float32, jnp.float32]


"""
Just gets the return for every episode without discriminating by task/level
"""
class FullReturnObserver(EnvLoopObserver):
  """docstring for LevelSuccessRateObserver"""
  def __init__(self):
    super(FullReturnObserver, self).__init__()

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    result = {
        f'0.full_return': self._episode_return,
    }
    return result

"""
In the MultiroomGoto environment, gives the return by room
"""
class RoomReturnObserver(EnvLoopObserver):
  """docstring for LevelReturnObserver"""
  def __init__(self):
    super(RoomReturnObserver, self).__init__()

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())
    self.room = env.env.room

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    result = {
        f'0.room/{self.room}/episode_return': self._episode_return,
    }
    return result