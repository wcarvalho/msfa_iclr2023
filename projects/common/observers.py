"""Evaluation Observers."""

import collections
import abc
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Sequence, Union
from acme.utils import signals
import os.path
from acme.utils.loggers.base import Logger
from acme.utils.observers import EnvLoopObserver
from acme.utils import paths

import dm_env
import pandas as pd
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
      # add latest (otherwise deleted)
      self.returns[self.level].append(self._episode_return)

      for key, returns in self.returns.items():
        avg = np.array(returns).mean()
        result[f'0.task/{key}/avg_return'] = float(avg)
        self.returns[key] = []

    return result


class EvalCountObserver(EnvLoopObserver):
  """Metric: Average return over many episodes"""
  def __init__(self, path, agent, seed, reset=1000, exit=False):
    super(EvalCountObserver, self).__init__()
    self.returns = collections.defaultdict(list)
    self.results = []
    self.level = None
    self.agent = agent
    self.seed = seed
    self.reset = reset
    self.exit = exit
    self.results_path = os.path.join(
        path,
        'evaluation',
        )
    if not os.path.exists(self.results_path):
      paths.process_path(self.results_path, add_uid=False)
    self.results_file = os.path.join(
          self.results_path,'eval_return_counts.csv')
    self.idx = 0


  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self.idx += 1

    if self.level is not None:
      self.add_prev_episode_to_results()
    self.level = str(env.env.current_levelname)

    # -----------------------
    # initialize episode return
    # -----------------------
    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())


    # -----------------------
    # initialize object counts
    # -----------------------
    self.pickup_counts = np.array(timestep.observation.observation.state_features)


  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

    self.pickup_counts = self.pickup_counts + np.array(timestep.observation.observation.state_features)

  def add_prev_episode_to_results(self):
    result=dict(
      episode_return=self._episode_return,
      agent=self.agent,
      seed=self.seed,
      level=self.level,
      episode_idx=self.idx,
      )
    for idx in range(len(self.pickup_counts)):
      result[f"object{idx}"] = self.pickup_counts[idx]

    self.results.append(result)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""

    with signals.runtime_terminator():
      if self.idx % self.reset == 0:
        df = pd.DataFrame(self.results)
        
        df.to_csv(self.results_file)

        if self.exit:
          import launchpad as lp  # pylint: disable=g-import-not-at-top
          lp.stop()
        self.results = []

    if self.results:
      return self.results[-1]
    else:
      return {}
