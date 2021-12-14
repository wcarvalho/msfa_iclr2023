"""Class used for MultiLevel version of Kitchen Env.

Each level can have different distractors, different layout,
    different tasks, etc. Very flexible since just takes in 
    dict(level_name:level_kwargs).
"""
import numpy as np

from gym import spaces


from gym_minigrid.minigrid import Grid, WorldObj
from babyai.levels.levelgen import RoomGridLevel, RejectSampling


from envs.babyai_kitchen.world import Kitchen
import envs.babyai_kitchen.tasks
from envs.babyai_kitchen.levelgen import KitchenLevel


class KitchenMultiLevel(object):

    """Wrapper environment that acts like the `current_level`.

    Everytime reset is called, a new level is sampled.

    Attributes:
        levelnames (list): names of levels
        levels (list): level objects
    """

    def __getattr__(self, name):
        """This is where all the magic happens. 

        This enables this class to act like `current_level`."""
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.current_level, name)

    def __init__(self, all_level_kwargs, **kwargs):

        # ======================================================
        # initialize levels
        # ======================================================
        self.levels = dict()
        for key, level_kwargs in all_level_kwargs.items():
            level_kwargs.update(kwargs)
            self.levels[key] = KitchenLevel(**level_kwargs)
        self.levelnames = list(self.levels.keys())


        self._current_idx = np.random.randint(len(self.levelnames))

    def add_level(self, obs):
        obs['level'] = self.current_levelname

    def reset(self, **kwargs):
        """Sample new level."""
        self._current_idx = np.random.randint(len(self.levelnames))
        obs = self.current_level.reset(**kwargs)
        self.add_level(obs)
        return obs

    def step(self, *args, **kwargs):
        """Sample new level."""
        obs, reward, done, info = self.current_level.step(*args, **kwargs)
        self.add_level(obs)
        return obs, reward, done, info

    @property
    def current_levelname(self):
        return self.levelnames[self._current_idx]

    @property
    def current_level(self):
        return self.levels[self.current_levelname]

