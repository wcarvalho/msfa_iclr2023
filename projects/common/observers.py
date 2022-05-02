"""Evaluation Observers."""

import collections
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


class LevelReturnObserver(EnvLoopObserver):
  """Metric: Return from Episode"""
  def __init__(self):
    super(LevelReturnObserver, self).__init__()

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())
    self.level = str(env.env.current_levelname)

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
        f'0.task/{self.level}/episode_return': self._episode_return,
    }
    return result

class LevelAvgReturnObserver(EnvLoopObserver):
  """Metric: Average return over many episodes"""
  def __init__(self, reset=200):
    super(LevelAvgReturnObserver, self).__init__()
    self.returns = collections.defaultdict(list)
    self.level = None
    self.reset = reset
    self.idx = 0


  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self.idx += 1
    if self.level is not None:
      self.returns[self.level].append(self._episode_return)

    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())
    self.level = str(env.env.current_levelname)


  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    result = {}

    if self.idx % self.reset == 0:
      for key, returns in self.returns.items():
        avg = np.array(returns).mean()
        result[f'0.task/{key}/avg_return'] = float(avg)
        self.returns[key] = []


    return result