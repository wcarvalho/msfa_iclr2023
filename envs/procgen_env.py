import numpy as np
import dm_env
from gym import spaces
from procgen import ProcgenEnv
from envs.gym_multitask import MultitaskGym

class ProcgenGymTask(object):
  """docstring for ProcgenGymTask"""
  def __init__(self, env, task,
    distribution_mode='easy',
    num_levels=200,
    ):
    super(ProcgenGymTask, self).__init__()
    print("="*50)
    print(f"Loading: {env}, {num_levels} levels")
    print("="*50)
    self._env = ProcgenEnv(
      env_name=env,
      num_envs=1,
      num_threads=1,
      distribution_mode=str(distribution_mode),
      num_levels=int(num_levels))

    self.task = np.array(task, dtype=np.float32)
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
    if info[0]['prev_level_complete']:
      import ipdb; ipdb.set_trace()
    # if done:
    #   import ipdb; ipdb.set_trace()
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

class ProcGenMultitask(MultitaskGym):
  """
  """

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    obs, reward, done, info = self.env.step(action)
    obs = self.ObsTuple(**{k: obs[k] for k in self.obs_keys})

    if info['prev_level_complete'] == 1:
      # finished level, done = False, but discount=0.0
      # avoids resetting environment
      timestep = dm_env.transition(reward=reward, observation=obs)
      timestep = timestep._replace(discount=0.0)
    else:
      if done:
        timestep = dm_env.termination(reward=reward, observation=obs)
      else:
        timestep = dm_env.transition(reward=reward, observation=obs)

    return timestep
