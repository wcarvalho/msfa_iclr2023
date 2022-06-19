# pylint: disable=g-bad-file-header
# Copyright 2019 The dm_env Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Catch reinforcement learning environment."""
from typing import NamedTuple

import dm_env
from dm_env import specs

from acme import types
from acme.wrappers import GymWrapper

import numpy as np

from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.goto_avoid import GotoAvoidEnv


class GenericObsTuple(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest


class MultitaskGeneric(dm_env.Environment):
  """
  """

  def __init__(self,
    all_level_kwargs: dict, 
    ObsTuple:GenericObsTuple=GenericObsTuple,
    LevelCls=GotoAvoidEnv,
    room_size=10,
    agent_view_size=5,
    path='.',
    tile_size=12,
    obs_keys = None,
    wrappers=None,
    **kwargs):
    """Initializes a Multitask environment environment.
    
    Args:
        all_level_kwargs (dict): initialization arguments for even task environment
        ObsTuple (GenericObsTuple, optional): Tuple used to create observations
        LevelCls (TYPE, optional): class used for each level
        room_size (int, optional): Description
        agent_view_size (int, optional): Description
        path (str, optional): root path from where script is run
        tile_size (int, optional): Description
        obs_keys (list, optional): keys for ObsTuple
        wrappers (None, optional): environment wrappers
        **kwargs: Description
    """

    self.env = MultiLevel(
        LevelCls=LevelCls,
        wrappers=wrappers,
        path=path,
        all_level_kwargs=all_level_kwargs,
        agent_view_size=agent_view_size,
        room_size=room_size,
        tile_size=tile_size,
        **kwargs)

    if wrappers:
      self.default_env = GymWrapper(self.env.env)
    else:
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
