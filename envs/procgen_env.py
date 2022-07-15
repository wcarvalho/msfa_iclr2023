import numpy as np
from gym import spaces
from procgen import ProcgenEnv

class ProcgenGymTask(object):
  """docstring for ProcgenGymTask"""
  def __init__(self, env, task,
    distribution_mode='easy',
    num_levels=200,
    ):
    super(ProcgenGymTask, self).__init__()
    print("="*50)
    print(f"Loading: {env}")
    print("="*50)
    self._env = ProcgenEnv(
      env_name=env,
      num_envs=1,
      num_threads=1,
      distribution_mode=str(distribution_mode),
      num_levels=int(num_levels))

    if len(task) == 3:
      self.task = np.array(task, dtype=np.float32)
    elif len(task) == 2:
      # reward for "present"
      self.task = np.array(task + [.1], dtype=np.float32)
    else:
      raise NotImplementedError
    # custom observation space
    image_space = self._env.observation_space['rgb']
    task_space = spaces.Box(
        low=0,
        high=255,
        shape=self.task.shape,
        dtype=self.task.dtype
    )
    self.observation_space = spaces.Dict({
        'image': image_space,
        'task': task_space,
    })


  def __getattr__(self, name):
    """This is where all the magic happens. 
    This enables this class to act like `env`."""
    return getattr(self._env, name)

  def step(self, action):
    image, reward, done, info = self._env.step(np.array((int(action),)))
    obs=dict(
      image=image['rgb'][0],
      task=self.task)
    return obs, reward[0], done[0], info[0]

  def reset(self):
    image = self._env.reset()
    obs=dict(
      image=image['rgb'][0],
      task=self.task)
    return obs
