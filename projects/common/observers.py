"""Evaluation Observers."""

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
  """docstring for LevelReturnObserver"""
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



# @dataclasses.dataclass
# class Trajectory:
#   goal: Any
#   # Uses the standard convention, [s0 a0 s1 a1 ... a{T-1} a{T}]
#   actions: List[np.ndarray] = dataclasses.field(default_factory=list)
#   observations: List[Any] = dataclasses.field(default_factory=list)
#   rewards: List[Number] = dataclasses.field(default_factory=list)

#   def add(
#       self, *,
#       action: Optional[np.ndarray],
#       observation: Any,
#       reward: Number,
#   ):  # yapf: disable
#     if action is not None:
#       self.actions.append(action)
#     self.observations.append(observation)
#     self.rewards.append(reward)


# class Metric(abc.ABC):

#   def reset(self):
#     """Reset and clear all the metrics stored (optionally).
#     Called every once at the beginning of the evaluation loop."""

#   @abc.abstractmethod
#   def result(self) -> Dict[str, Any]:
#     """Return the aggregated result over several episodes.
#     The result can be either metrics (dict) or some artifacts to save
#     (video, plot, etc.) where key represents their name or label.
#     """
#     # TODO: What if result() is called before compute_metric is called?
#     raise NotImplementedError

#   def __str__(self) -> str:
#     return type(self).__name__


# class StepwiseMetric(Metric):

#   @abc.abstractmethod
#   def on_step(
#       self,
#       env,
#       timestep: dm_env.TimeStep,
#       action: Optional[np.ndarray] = None,
#   ):
#     # Note: timestep has GoalEnv spec.
#     raise NotImplementedError

#   # TODO: How to notify episode has finished?


# class EpisodeMetric(Metric):
#   """A Metric that works on an episode-level."""

#   @abc.abstractmethod
#   def compute_episode_metric(
#       self,
#       trajectory: Trajectory,
#   ) -> Dict[str, Number]:
#     """Compute metric for a single episode/trajectory.
#     Called when an episode is complete.
#     """
#     del trajectory  # unused
#     raise NotImplementedError


# class LevelReturnObserver(EnvLoopObserver):
#   """An environment loop observer for evaluation."""

#   _current_episode: Trajectory
#   _step_metrics: Sequence[StepwiseMetric]
#   _episode_metrics: Sequence[EpisodeMetric]

#   def __init__(
#       self,
#       *,  # yapf: disable
#       step_metrics: Sequence[StepwiseMetric] = (),
#       episode_metrics: Sequence[EpisodeMetric] = (),
#       # logger: Logger,
#       # artifacts_path: str,
#   ):
#     self._current_episode = None  # type: ignore
#     self._step_metrics = tuple(step_metrics)
#     self._episode_metrics = tuple(episode_metrics)
#     self._logger = logger
#     self._artifacts_path = artifacts_path

#   def observe_first(self, env, timestep: dm_env.TimeStep):
#     import ipdb; ipdb.set_trace()
#     # TODO: We assumed timestep came from GoalEnv, but this may not be true.
#     self._current_episode = Trajectory(goal=timestep.observation.desired_goal)
#     self._current_episode.add(
#         action=None,
#         observation=timestep.observation.observation,
#         reward=timestep.reward,
#     )

#     for m in self._step_metrics:
#       m.on_step(env, timestep, action=None)

#   def observe(self, env, timestep: dm_env.TimeStep, action: np.ndarray):
#     import ipdb; ipdb.set_trace()
#     # TODO: We assumed timestep came from GoalEnv, but this may not be true.
#     self._current_episode.add(
#         action=action,
#         observation=timestep.observation.observation,
#         reward=timestep.reward)

#     for m in self._step_metrics:
#       m.on_step(env, timestep, action=action)

#   def get_metrics(self) -> Dict[str, Number]:
#     # Called at the end of the episode, in the EnvironmentLoop
#     result = {}
#     for m in self._episode_metrics:
#       import ipdb; ipdb.set_trace()
#       result.update(m.compute_episode_metric(self._current_episode))
#     return result

#   def reset(self):
#     import ipdb; ipdb.set_trace()
#     for m in self._episode_metrics:
#       m.reset()
#     for m in self._step_metrics:
#       m.reset()

#   def write_results(self, metrics: Dict[str, Number], *, steps: int):
#     """Report, and save all the evaluation results.
#     This is called after a several number of evaluation episodes are all
#     finished, so usually used for saving aggregated results."""

#     # Write the (aggregated) metrics from EnvironmentLoop and etc
#     # for this evaluation iteration.
#     self._logger.write(metrics)

#     # Report, write, or save other metrics
#     log_data = {}
#     import ipdb; ipdb.set_trace()
#     for m in itertools.chain(self._episode_metrics, self._step_metrics):
#       ret = m.result()
#       # TODO this hardcoded name may not work. evalu
#       # This is needed for what?
#       # This must match steps_key in eval_logger(...)
#       ret['steps'] = steps  # global_step
#       ret['evaluator_steps'] = steps  # global_step
#       assert isinstance(ret, dict), str(type(ret))

#       log_data.update(ret)
#       print("-", m, ":", ret)  # TODO pretty print, etc.

#     self._logger.write(log_data)