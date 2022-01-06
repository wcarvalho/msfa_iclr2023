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
"""Kitchen reinforcement learning environment."""
from typing import NamedTuple

import dm_env
from dm_env import specs

from acme import types
from acme.wrappers import GymWrapper

import numpy as np

from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.levelgen import KitchenLevel

class Observation(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest

def convert_rawobs(obs):
    obs.pop('mission_idx')
    obs['image'] = obs['image'] / 255.0
    return Observation(**obs)

class MultitaskKitchen(dm_env.Environment):
  """
  """

  def __init__(self,
    tasks: list,
    room_size=10,
    agent_view_size=5,
    path='.',
    tile_size=12,
    wrappers=None,
    num_dists=0,
    **kwargs):
    """Initializes a new Kitchen environment.
    Args:
      rows: number of rows.
      columns: number of columns.
      seed: random seed for the RNG.
    """
    all_level_kwargs = dict()
    for task in tasks:
        all_level_kwargs[task]=dict(
            room_size=room_size,
            agent_view_size=agent_view_size,
            task_kinds=[task],
            tile_size=tile_size,
            num_dists=num_dists,
        )

    self.env = MultiLevel(
        LevelCls=KitchenLevel,
        wrappers=wrappers,
        path=path,
        all_level_kwargs=all_level_kwargs,
        **kwargs)


    self.default_env = GymWrapper(self.env.env)


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    obs = self.env.reset()
    obs = convert_rawobs(obs)
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    obs = convert_rawobs(obs)

    if done:
      return dm_env.termination(reward=reward, observation=obs)
    else:
      return dm_env.transition(reward=reward, observation=obs)


  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return self.default_env.action_spec()

  def observation_spec(self):
    default = self.default_env.observation_spec()
    return Observation(
        image=specs.BoundedArray(
            shape=default['image'].shape,
            dtype=np.float32,
            name="image",
            minimum=0,
            maximum=1,
        ),
        mission=default['mission'])
