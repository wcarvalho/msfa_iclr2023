
"""Multiroom_goto env wrapper"""
from typing import NamedTuple

import dm_env
from dm_env import specs

from acme import types
from acme.wrappers import GymWrapper

import numpy as np

from envs.babyai_kitchen.multilevel import MultiLevel
from envs.babyai_kitchen.multiroom_goto import MultiroomGotoEnv

# from envs.acme.babyai import BabyAI


class GotoObs(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  pickup: types.Nest
  image: types.Nest
  mission: types.Nest


class MultiroomGoto(dm_env.Environment):

  def __init__(self,
               objectlists,
               tile_size=8,
               rootdir='.',
               verbosity=0,
               pickup_required=True,
               epsilon=0.0,
               room_size=8,
               doors_start_open=False,
               stop_when_gone=False,
               wrappers=None,
               walls_gone=False,
    **kwargs):
    """Initializes a new MultiroomGotoEnv

      Args:
          *args: Description
          objectlists (TYPE): Each object list is a nested list of [{object_name: object_quantity}] one dictionary for each of the three rooms.
          We have many of these, one for each level, in dict form
          tile_size (int, optional): how many pixels to use for a tile, I think
          rootdir (str, optional): Just a path for the kitchen env to search for files, probably leave this be
          verbosity (int, optional): how much to print

          epsilon (float, optional): chance that an object is not in its usual room
          room_size (int, optional): the size of a room, duh
          doors_start_open (bool, optional): make the doors start open (default is closed but unlocked)
          stop_when_gone (bool, optional): should we stop the episode when all the objects with reward associated are gone?
          **kwargs: Description
      """
    all_level_kwargs = dict()
    for key, objectlist in objectlists.items():
        all_level_kwargs[key]=dict(
            objectlist=objectlist,
            tile_size=tile_size,
            rootdir=rootdir,
            verbosity=verbosity,
            pickup_required=pickup_required,
            epsilon=epsilon,
            room_size=room_size,
            doors_start_open=doors_start_open,
            stop_when_gone=stop_when_gone,
            walls_gone=walls_gone
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

    self.keys = ['pickup','image' , 'mission']


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
