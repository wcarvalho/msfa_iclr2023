from acme.wrappers import base
import copy
import dm_env
import tree
from acme import types
import numpy as np
from collections import namedtuple


from envs.babyai_kitchen import tasks as kitchen_tasks

def reset_tasks(tasks: dict):
  [t.reset() for t in tasks.values()]

def task_names(tasks: dict):
  return [t.instruction for t in tasks.values()]

class TrainTasksWrapper(base.EnvironmentWrapper):
  """A wrapper that adds train tasks."""

  def __init__(self, *args, train_tasks, task_reps, instr_preproc, max_length, reset_at_step=False, **kwargs):
    super().__init__(*args, **kwargs)

    self.reset_at_step = reset_at_step
    self.instr_preproc = instr_preproc
    self.max_length = max_length

    all_tasks = kitchen_tasks.all_tasks()
    kitchen_copy = copy.deepcopy(self._environment.default_gym.kitchen)
    # env only needed on resets
    self.train_tasks = {t:all_tasks[t](env=None, kitchen=kitchen_copy, task_reps=task_reps) for t in train_tasks}

    # -----------------------
    # obs stuff
    # -----------------------
    original_obs_spec = self._environment.observation_spec()
    self.original_obs_dict = original_obs_spec._asdict()
    self.tasks_shape = (len(train_tasks), max_length)
    self.original_obs_dict['train_tasks'] = dm_env.specs.BoundedArray(
        shape=self.tasks_shape,
        dtype=np.uint8,
        name="train_tasks",
        minimum=np.zeros(self.tasks_shape),
        maximum=np.ones(self.tasks_shape),
    )
    self.keys = list(self.original_obs_dict.keys())

    self.Obs = namedtuple('TasksObs', self.keys)
    self._obs_spec = self.Obs(**self.original_obs_dict)

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    reset_tasks(self.train_tasks)
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    if self.reset_at_step:
      import ipdb; ipdb.set_trace()
      reset_tasks(self.train_tasks)
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Add train tasks to observation"""
    train_tasks = task_names(self.train_tasks)
    tasks_array = np.zeros(self.tasks_shape, dtype=np.uint8)
    for idx, t in enumerate(train_tasks):
      arr = self.instr_preproc(t)
      tasks_array[idx, :len(arr)] = arr

    obs = timestep.observation._asdict()
    obs['train_tasks'] = tasks_array
    new_timestep = timestep._replace(observation=self.Obs(**obs))

    return new_timestep

  def observation_spec(self):
    """Remap keys in obs spec. created named tuple.
    """
    return self._obs_spec


