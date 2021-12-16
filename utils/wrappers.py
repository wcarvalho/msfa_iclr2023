from acme.wrappers import base
import dm_env
import tree
from acme import types

class ObservationRemapWrapper(base.EnvironmentWrapper):
  """A wrapper that remaps names of observation fields."""

  def __init__(self, *args, remap=None, **kwargs):
    super().__init__(*args, **kwargs)
    self._remap = remap
    assert remap is not None

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    import ipdb; ipdb.set_trace()

    return timestep._replace(observation=oar)

