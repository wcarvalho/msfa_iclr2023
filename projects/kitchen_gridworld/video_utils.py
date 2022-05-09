
import operator
import time
from typing import Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import tree
from acme.environment_loop import EnvironmentLoop

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


import pickle
from acme import core
from acme import types
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import networks as r2d2_networks_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
# from acme.jax import utils
from acme.jax.types import ModelToSnapshot

import haiku as hk
import jax
import rlax
import numpy as np

from agents.td_agent.types import TDNetworkFns, Predictions
from agents.td_agent.configs import R2D1Config


class DataStorer(object):
  """Metric: Average return over many episodes"""
  def __init__(self, path, agent, seed, episodes=20, exit=False):
    super(DataStorer, self).__init__()
    self.all_episode_data = collections.defaultdict(list)
    self.level = None
    self.agent = agent
    self.seed = seed
    self.episodes = episodes
    self.exit = exit
    self.results_path = os.path.join(
        path,
        'episode_data',
        )
    if not os.path.exists(self.results_path):
      paths.process_path(self.results_path, add_uid=False)
    
    self.idx = 0

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Record level and store previous data."""
    if self.level is not None:
      self.add_prev_episode_to_results()
    
    self.level = str(env.env.current_levelname)
    self.episode_data = []
    self.interaction_info = []
    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())


  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Skip"""
    info = env.env.interaction_info
    self.interaction_info.append(info)
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

  def store(self, data) -> None:
    """Records one environment step."""
    self.episode_data.append(data)

  def add_prev_episode_to_results(self):
    """
    1. convert data to numpy
    2. stack arrays by key (e.g. obs, q, etc.)
    3. agglomerate interaction info + reward
    """
    processed_episode_data = jax.tree_map(np.array, *self.episode_data)
    # turn in stacked numpy arrays
    processed_episode_data = jax.tree_map(lambda *arrays: np.stack(arrays), processed_episode_data)
    processed_episode_data['interaction_info'] = self.interaction_info
    processed_episode_data['return'] = self._episode_return

    self.all_episode_data[self.level].append(processed_episode_data)
    with signals.runtime_terminator():
      if self.idx % self.episodes == 0:
        path = os.path.join(self.results_path, 'data.npz')
        with open(path, 'wb') as file:
          pickle.dump(self.episode_data, file)

        if self.exit:
          import launchpad as lp  # pylint: disable=g-import-not-at-top
          lp.stop()
        self.episode_data = []


  def get_metrics(self):
    """Returns metrics collected for the current episode."""

    return {}


def make_behavior_policy(
    networks: TDNetworkFns,
    config: R2D1Config,
    evaluation: bool = False,
    ) -> r2d2_networks_lib.EpsilonRecurrentPolicy:
  """Selects action according to the policy.
  
  Args:
      networks (TDNetworkFns): Network functions
      config (R2D1Config): Config
      evaluation (bool, optional): whether evaluation policy
      network_samples (bool, optional): whether network is random
  
  Returns:
      r2d2_networks_lib.EpsilonRecurrentPolicy: epsilon-greedy policy
  """

  def behavior_policy(
                      params: networks_lib.Params,
                      key: networks_lib.PRNGKey,
                      observation: types.NestedArray,
                      core_state: types.NestedArray,
                      epsilon):
    key, key_net, key_sample = jax.random.split(key, 3)

    # -----------------------
    # if evaluating & have seperation evaluation function, use it
    # -----------------------
    if evaluation:
      if networks.evaluation is not None:
        forward_fn = networks.evaluation.apply
      else:
        forward_fn = networks.forward.apply
    else:
      forward_fn = networks.forward.apply

    preds, core_state = forward_fn(
        params, key_net, observation, core_state, key_sample)


    return preds, core_state

  return behavior_policy


class ActorStorageWrapper(object):
  """docstring for ActorStorageWrapper"""
  def __init__(self, agent , observer : DataStorer, epsilon : float, seed : int):
    """Summary
    
    Args:
        agent (TYPE): Description
        observer (DataStorer): Description
        epsilon (float): Description
        seed (int): Description
    """
    super(ActorStorageWrapper, self).__init__()
    self.agent = agent
    self.observer = observer
    self.epsilon = epsilon
    self.rng = jax.random.PRNGKey(seed)

  def __getattr__(self, name):
    return getattr(self.agent, name)

  def select_action(self,
                    observation: networks_lib.Observation) -> types.NestedArray:
    preds, self.agent._state = self._policy(self._params, observation, self._state)

    self.rng, sample_rng = jax.random.split(self.rng, 2)
    action = rlax.epsilon_greedy(self.epsilon).sample(sample_rng, preds.q)

    # -----------------------
    # store
    # -----------------------
    self.observer.store(utils.to_numpy(dict(
          observation=observation,
          action=action,
          preds=preds,
          lstm_state=self.agent._state.recurrent_state)))

    return utils.to_numpy(action)

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

