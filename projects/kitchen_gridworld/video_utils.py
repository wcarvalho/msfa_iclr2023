
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
from acme import specs as acme_specs
from dm_env import specs
import jax.numpy as jnp
import numpy as np
import operator
import tree


import cloudpickle
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
from agents.td_agent.agents import DistributedTDAgent


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
    self.results_file = os.path.join(self.results_path, 'data.npz')
    
    self.idx = 0

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Record level and store previous data."""
    if self.level is not None:
      self.add_prev_episode_to_results()
    
    self.idx += 1
    self.level = str(env.env.current_levelname)
    self.episode_data = []
    self.interaction_info = []
    self.rewards = []
    # self._episode_return = tree.map_structure(
    #   _generate_zeros_from_spec,
    #   env.reward_spec())


  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Skip"""
    info = env.env.interaction_info
    self.interaction_info.append(info)
    self.rewards.append(timestep.reward)
    # self._episode_return = tree.map_structure(
    #   operator.iadd,
    #   self._episode_return,
    #   timestep.reward)

  def store(self, data) -> None:
    """Records one environment step."""
    self.episode_data.append(data)

  def add_prev_episode_to_results(self):
    """
    1. convert data to numpy
    2. stack arrays by key (e.g. obs, q, etc.)
    3. agglomerate interaction info + reward
    """
    # [{key:data}, {key:data}, ...]
    episode_data = utils.to_numpy(self.episode_data)

     # {key: [data, data, ...]}
    episode_data_dict = jax.tree_map(lambda *arrays: np.stack(arrays), *episode_data)
    
    # add 2 keys
    episode_data_dict['interaction_info'] = self.interaction_info
    episode_data_dict['rewards'] = np.array(self.rewards)

    # fix observation data
    obs = episode_data_dict['observation'].observation
    episode_data_dict['observation'] = obs._asdict()

    self.all_episode_data[self.level].append(episode_data_dict)
    with signals.runtime_terminator():
      if self.idx % (self.episodes+1) == 0:
        with open(self.results_file, 'wb') as file:
          cloudpickle.dump(self.all_episode_data, file)

        if self.exit:
          import launchpad as lp  # pylint: disable=g-import-not-at-top
          lp.stop()
          import os; os._exit(0)
        self.all_episode_data = collections.defaultdict(list)


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

def get_cumulants(unroll_fn, params, prior_observation, observation, state, rng):

  rng, key_rng = jax.random.split(rng, 2)

  # [T=2, B=1, D, ...]
  both_obs = jax.tree_map(lambda *arrs: jnp.stack(arrs), *(
    utils.tile_nested(prior_observation, 1),
    utils.tile_nested(observation, 1)))

  # [B=1, D, ...]
  state = utils.tile_nested(state, 1)


  # jax.tree_map(lambda x:x.shape, both_obs)
  # jax.tree_map(lambda x:x.shape, state)
  preds, unroll_state = unroll_fn.apply(params, rng, both_obs, state.recurrent_state, key_rng)
  return preds.cumulants[0,0]



class ActorStorageWrapper(object):
  """docstring for ActorStorageWrapper"""
  def __init__(self, agent , observer : DataStorer, epsilon : float, seed : int, networks=None):
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
    self.networks = networks
    self.rng = jax.random.PRNGKey(seed)

  def __getattr__(self, name):
    return getattr(self.agent, name)

  def observe_first(self, timestep: dm_env.TimeStep):
    """In first time-step, get empty observation. Needed for cumulants.
    """
    self.prior_obs = utils.zeros_like(timestep.observation)
    self.agent.observe_first(timestep)

  def select_action(self,
                    observation: networks_lib.Observation) -> types.NestedArray:

    state_pre_update = self._state
    preds, self.agent._state = self._policy(self._params, observation, self._state)

    self.rng, sample_rng = jax.random.split(self.rng, 2)
    action = rlax.epsilon_greedy(self.epsilon).sample(sample_rng, preds.q)


    computed = dict(
          observation=observation,
          action=action,
          preds=preds,
          lstm_state=self.agent._state.recurrent_state)
    if self.networks is not None:
      computed['cumulants'] = get_cumulants(
        unroll_fn=self.networks.unroll,
        params=self._params,
        prior_observation=self.prior_obs, # from timestep t-1
        observation=observation, # from timestep t
        state=state_pre_update,
        rng=sample_rng)

    # -----------------------
    # store
    # -----------------------
    self.observer.store(computed)
    self.prior_obs = observation

    return utils.to_numpy(action)

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)

def storage_evaluator_factory(
    environment_factory,
    network_factory,
    policy_factory,
    epsilon: float,
    seed: int,
    observers,
    log_to_bigtable,
    logger_fn=None,
    ):
  """Returns a default evaluator process."""
  def evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))
    networks = network_factory(acme_specs.make_environment_spec(environment))

    actor = make_actor(actor_key, policy_factory(networks), variable_source)
    actor = ActorStorageWrapper(
      agent=actor,
      observer=observers[0],
      epsilon=epsilon,
      seed=seed)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    if logger_fn is not None:
      logger = logger_fn('evaluator', 'actor_steps')
    else:
      logger = loggers.make_default_logger(
          'evaluator', log_to_bigtable, steps_key='actor_steps')

    # Create the run loop and return it.
    return EnvironmentLoop(environment, actor, counter,
                                            logger, observers=observers)
  return evaluator