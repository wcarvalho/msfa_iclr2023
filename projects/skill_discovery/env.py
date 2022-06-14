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

from envs.babyai.multilevel import MultiLevel
from envs.babyai_kichen.levelgen import KitchenLevel

class BabyAiObs(NamedTuple):
  image: types.Nest
  mission: types.Nest


class BabyAISkills(dm_env.Environment):
  """
  """

  def __init__(self,
    all_level_kwargs: dict, 
    room_size=8,
    agent_view_size=5,
    wrappers=None,
    LevelCls=KitchenLevel,
    ObsCreator=BabyAiObs,
    obs_keys=['image', 'mission'],
    **kwargs):
    """Initializes a new Goto/Avoid environment.
    
    Args:
        obj2rew (dict): Description
        room_size (int, optional): Description
        agent_view_size (int, optional): Description
        wrappers (None, optional): Description
        LevelCls (TYPE, optional): Description
        **kwargs: Description

    """
    for key in all_level_kwargs.keys():
      all_level_kwargs[key].update(dict(
          room_size=room_size,
          agent_view_size=agent_view_size,
      ))

    self.env = MultiLevel(
        LevelCls=LevelCls,
        wrappers=wrappers,
        all_level_kwargs=all_level_kwargs,
        **kwargs)

    if wrappers:
      self.default_env = GymWrapper(self.env.env)
    else:
      self.default_env = GymWrapper(self.env)

    self.ObsCreator = ObsCreator
    self.obs_keys = obs_keys


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    obs = self.env.reset()
    obs = self.ObsCreator(**{k: obs[k] for k in self.obs_keys})
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    reward = float(reward > 0.0) # 1 if complete, 0 otherwise

    obs = self.ObsCreator(**{k: obs[k] for k in self.obs_keys})

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
    return self.ObsCreator(**default)
