from acme.wrappers import base
import dm_env
import tree
from acme import types

from collections import namedtuple

class ObservationRemapWrapper(base.EnvironmentWrapper):
  """A wrapper that remaps names of observation fields."""

  def __init__(self, *args, remap=None, **kwargs):
    super().__init__(*args, **kwargs)
    assert remap is not None

    # obs spec
    original_obs_spec = self._environment.observation_spec()
    self.original_obs_dict = original_obs_spec._asdict()

    # create full mapping
    okeys = self.original_obs_dict.keys()
    self.original_fields = set(okeys)
    self.replace_fields = set(remap.keys())
    self.keep_fields = self.original_fields - self.replace_fields

    # self.new_fields = self.keep_fields.union(set(remap.keys()))
    self.remap = {k:k for k in self.keep_fields}
    self.remap.update(remap)

    # new obs spec
    self.Obs = namedtuple('Observation', [self.remap[o] for o in okeys])


  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def _remap(self, old):
    """Remap kyes in dict"""
    new = {}
    for oldk, newk in self.remap.items():
        new[newk] = old[oldk]
    return new

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    obs = self._remap(timestep.observation._asdict())
    return timestep._replace(observation=self.Obs(**obs))

  def observation_spec(self):
    """Remap keys in obs spec. created named tuple.
    """
    new_spec = self._remap(self.original_obs_dict)
    return self.Obs(**new_spec)

