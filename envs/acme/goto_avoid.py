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

import dm_env
from dm_env import specs
from acme.wrappers import GymWrapper
import numpy as np

_ACTIONS = (0, 1, 3, 4)  # Left, right, forward, pickup

from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.goto_avoid import GotoAvoidEnv

class GoToAvoid(dm_env.Environment):
  """
  """

  def __init__(self,
    obj2rew: dict, 
    room_size=10,
    partial=True,
    agent_view_size=5,
    tile_size=12,
    wrappers=None,
    nobjects=10):
    """Initializes a new Catch environment.
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
        )

    self.env = MultiLevel(
        LevelCls=GotoAvoidEnv,
        wrappers=wrappers,
        all_level_kwargs=all_level_kwargs)


    self.default_env = GymWrapper(self.env.env)
    self.ex_rewards = next(iter(obj2rew.values()))
    self.ntasks = len(obj2rew)

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    # self._reset_next_step = False
    # self._ball_x = self._rng.randint(self._columns)
    # self._ball_y = 0
    # self._paddle_x = self._columns // 2
    # return dm_env.restart(self._observation())
    import ipdb; ipdb.set_trace()

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    import ipdb; ipdb.set_trace()
    # if self._reset_next_step:
    #   return self.reset()

    # # Move the paddle.
    # dx = _ACTIONS[action]
    # self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

    # # Drop the ball.
    # self._ball_y += 1

    # # Check for termination.
    # if self._ball_y == self._paddle_y:
    #   reward = 1. if self._paddle_x == self._ball_x else -1.
    #   self._reset_next_step = True
    #   return dm_env.termination(reward=reward, observation=self._observation())
    # else:
    #   return dm_env.transition(reward=0., observation=self._observation())

  def observation_spec(self) -> specs.BoundedArray:
    """Returns the observation spec."""
    return self.default_env.observation_spec()

  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return self.default_env.action_spec()
