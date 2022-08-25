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

# from envs.acme.babyai import BabyAI


class GotoObs(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  pickup: types.Nest
  mission: types.Nest


class GoToAvoid(dm_env.Environment):
  """
  """

  def __init__(self,
    obj2rew: dict, 
    room_size=10,
    agent_view_size=5,
    path='.',
    tile_size=12,
    wrappers=None,
    nobjects=10,
    respawn=False,
    ObsCls=GotoObs,
    **kwargs):
    """Initializes a new Goto/Avoid environment.
    Args:
      rows: number of rows.
      columns: number of columns.
      seed: random seed for the RNG.
    """
    all_level_kwargs = dict()
    for key, o2r in obj2rew.items():
        all_level_kwargs[key]=dict(
            room_size=room_size,
            agent_view_size=agent_view_size,
            object2reward=o2r,
            tile_size=tile_size,
            nobjects=nobjects,
            respawn=respawn,
            objects=list(o2r.keys()),
        )

    self.ObsCls = ObsCls
    self.env = MultiLevel(
        LevelCls=GotoAvoidEnv,
        wrappers=wrappers,
        path=path,
        all_level_kwargs=all_level_kwargs,
        **kwargs)

    if wrappers:
      self.default_env = GymWrapper(self.env.env)
    else:
      self.default_env = GymWrapper(self.env)

    self.keys = ['image', 'pickup', 'mission']


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    obs = self.env.reset()
    obs = self.ObsCls(**{k: obs[k] for k in self.keys})
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    obs = self.ObsCls(**{k: obs[k] for k in self.keys})

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
    return self.ObsCls(**default)
