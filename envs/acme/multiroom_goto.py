
"""Multiroom_goto env wrapper"""
from typing import NamedTuple

import dm_env
from dm_env import specs

from acme import types
from acme.wrappers import GymWrapper
import functools

import numpy as np

from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.multiroom_goto import MultiroomGotoEnv

# from envs.acme.babyai import BabyAI


class GotoObs(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest
  pickup: types.Nest
  train_tasks: types.Nest


class MultiroomGoto(dm_env.Environment):

  def __init__(self,
               objectlist,
               mission_objects = None,
               tile_size=8,
               rootdir='.',
               verbosity=0,
               pickup_required=True,
               epsilon=0.0,
               room_size=8,
               doors_start_open=False,
               stop_when_gone=False,
               walls_gone=False,
               one_room = False,
               two_rooms = False,
               deterministic_rooms = False,
               room_reward_task_vector = True,
               room_reward = 0.0,
               wrappers=None,
    **kwargs):
    """Initializes a new MultiroomGotoEnv

      Args:
          See the babyai_kitchen/multiroom_goto.py file for an explanation of arguments related to the env
          The only unique arguments here are:
          mission_objects (Optional): This should just be a list of strings corresponding to all the objects in the env
            which you want to have tasks associated with. Each mission object will then be assigned to its own "level"
            for logging purposes. Highly recommended you use this instead of leaving it as None
          wrappers (Optional) which allows you to add env wrappers to the environment
          **kwargs: Description

          Everything else is totally boilerplate!
      """

    all_level_kwargs = dict()
    if mission_objects:
        for mission_object in mission_objects:
            all_level_kwargs[mission_object] = dict(
                objectlist=objectlist,
                mission_object=mission_object,
                tile_size=tile_size,
                rootdir=rootdir,
                verbosity=verbosity,
                pickup_required=pickup_required,
                epsilon=epsilon,
                room_size=room_size,
                doors_start_open=doors_start_open,
                stop_when_gone=stop_when_gone,
                walls_gone=walls_gone,
                one_room=one_room,
                two_rooms = two_rooms,
                deterministic_rooms = deterministic_rooms,
                room_reward=room_reward,
                room_reward_task_vector = room_reward_task_vector
            )
    else:
        all_level_kwargs['ONLY_LEVEL']  = dict(
                objectlist=objectlist,
                tile_size=tile_size,
                rootdir=rootdir,
                verbosity=verbosity,
                pickup_required=pickup_required,
                epsilon=epsilon,
                room_size=room_size,
                doors_start_open=doors_start_open,
                stop_when_gone=stop_when_gone,
                walls_gone=walls_gone,
                one_room=one_room,
                two_rooms = two_rooms,
                deterministic_rooms = deterministic_rooms,
                room_reward=room_reward,
                room_reward_task_vector=room_reward_task_vector
            )


    self.env = MultiLevel(
        LevelCls=MultiroomGotoEnv,
        wrappers=wrappers,
        path=rootdir,
        all_level_kwargs=all_level_kwargs,
        **kwargs)

    if wrappers:
      self.default_env = GymWrapper(self.env.env)
    else:
      self.default_env = GymWrapper(self.env)

    self.keys = sorted(['image', 'pickup' , 'mission', 'train_tasks']) #we sort these to work with the env remap wrapper


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    obs = self.env.reset()
    obs = GotoObs(**{k: obs[k] for k in self.keys})
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    obs = GotoObs(**{k: obs[k] for k in self.keys})

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
    return GotoObs(**default)
