"""
python -m ipdb -c continue envs/babyai_kitchen/goto_avoid.py
"""

from gym import spaces

import collections
import math
import numpy as np
import copy
from envs.babyai_kitchen.world import Kitchen
from envs.babyai_kitchen.levelgen import KitchenLevel

def reject_next_to(env, pos):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """

    sx, sy = env.agent_pos
    x, y = pos
    d = abs(sx - x) + abs(sy - y)
    return d < 2

class KitchenComboLevel(KitchenLevel):
  """docstring for KitchenComboLevel"""
  def __init__(self, 
    *args,
    task2reward,
    tile_size=8,
    rootdir='.',
    verbosity=0,
    ntasks=1,
    kitchen=None,
    objects=None,
    # pickup_required=True,
    **kwargs):
    """Summary
    
    Args:
        *args: Description
        task2reward (TYPE): Description
        tile_size (int, optional): Description
        rootdir (str, optional): Description
        verbosity (int, optional): Description
        ntasks (int, optional): How many times to spawn each objects type
        kitchen (None, optional): Description
        **kwargs: Description
    """
    self.task2reward = task2reward
    # self._task_objects = [k for k,v in task2reward.items() if v > 0]
    # self.pickup_required = pickup_required

    self.mission_arr = np.array(
        list(self.task2reward.values()),
        dtype=np.int32,
        )
    self.train_tasks = np.identity(len(self.mission_arr), dtype=np.int32,)

    assert (self.mission_arr >= 0).all()
    # self.object_names = list(task2reward.keys())
    # if objects:
    #   assert objects == self.object_names
    # else:
    #   objects = self.object_names

    # self.type2idx = {o:idx for idx, o in enumerate(objects)}
    # self._task_oidxs = [self.type2idx[o] for o in self._task_objects]
    self.ntasks = ntasks
    kitchen = kitchen or Kitchen(
        # objects=objects,
        tile_size=tile_size,
        rootdir=rootdir,
        verbosity=verbosity,
        )

    # self.default_objects = []
    # for _ in range(nobjects):
    #   self.default_objects.extend(copy.deepcopy(kitchen.objects))
    

    # kwargs["task_kinds"] = ['pickup']
    # kwargs['actions'] = ['left', 'right', 'forward', 'pickup_contents']
    # kwargs['kitchen'] = kitchen
    super().__init__(
        *args,
        tile_size=tile_size,
        rootdir=rootdir,
        verbosity=verbosity,
        objects=objects,
        kitchen=kitchen,
        **kwargs)
    # -----------------------
    # backwards compatibility
    # -----------------------
    # self.actions.drop = len(self.actions)
    # self.actions.toggle = len(self.actions)

    self.observation_space.spaces['mission'] = spaces.Box(
        low=0,
        high=255,
        shape=(len(self.task2reward),),
        dtype='int32'
    )
    self.observation_space.spaces['train_tasks'] = spaces.Box(
        low=0,
        high=255,
        shape=(len(self.task2reward), len(self.task2reward)),
        dtype='int32'
    )

  # @property
  # def task_objects(self):
  #   return self._task_objects

  def generate_task(self):

    # connect all rooms
    self.connect_all()

    self.kitchen.reset(randomize_states=self.random_object_state)

    self.task2checkers =[]
    # -----------------------
    # get task objects to spawn
    # -----------------------
    task_object_types = []
    task_objects = set()
    for task_kind, reward in self.task2reward.items():
      for task_idx in range(self.ntasks):
        # make task cheker
        checker = self.rand_task(task_kind, init=False)
        self.task2checkers.append(checker)

        # generate task
        checker.reset(exclude=task_object_types)
        task_object_types.extend(checker.task_types)
        task_objects.update(checker.task_objects)

    # -----------------------
    # spawn objects
    # -----------------------
    # self.object_occurrences = np.zeros(len(self.task2reward), dtype=np.int32)
    for object in task_objects:
        # object_type = object.type
        # object_idx = self.type2idx[object_type]
        # self.object_occurrences[object_idx] += 1
        # object = self.default_objects[object_idx]
        # assert object.type in self.task2reward, "placing object with no reward signature"
        self.place_in_room(0, 0, object)

    # self.remaining = np.array(self.object_occurrences)
    # The agent must be placed after all the object to respect constraints
    while True:
        self.place_agent()
        start_room = self.room_from_pos(*self.agent_pos)
        # Ensure that we are not placing the agent in the locked room
        if start_room is self.locked_room:
            continue
        break
    self.check_objs_reachable()

  def reset(self):
    obs = super().reset()
    assert self.carrying is None
    # obs['pickup'] = np.zeros(len(self.task2reward), dtype=np.int32)
    obs['mission'] = self.mission_arr
    obs['train_tasks'] = self.train_tasks

    return obs

  def reset_checker(self, checker):
    # get reward
    # object = self.grid.get(*fwd_pos)

    objects = checker.task_objects

    for object in objects:

      # reset position
      pos = object.cur_pos
      if (pos >= 0).all():
        self.grid.set(*pos, None)
      object.reset()

      # move object
      room = self.get_room(0, 0)
      pos = self.place_obj(
          object,
          room.top,
          room.size,
          reject_fn=reject_next_to,
          max_tries=1000
        )

    # print('resetting', checker)
    checker.reset_task()
    self.carrying = None
    self.kitchen.update_carrying(None)


  def step(self, action):
    obs, total_reward, done, info = super().step(action)

    for checker in self.task2checkers:
      reward, task_done = checker.get_reward_done()
      total_reward += float(reward)

      if task_done:
        checker.terminate()
      checker.increment_done()

      # wait 1 time-step to observe eff
      if checker.time_complete > 1:
        self.reset_checker(checker)
    # print('action', self.idx2action[action])
    obs['mission'] = self.mission_arr
    obs['train_tasks'] = self.train_tasks
    return obs, total_reward, done, info


  @property
  def total_remaining(self):
    return self.remaining[self._task_oidxs].sum()

if __name__ == '__main__':
    import gym_minigrid.window
    import time
    from envs.babyai_kitchen.bot import GotoAvoidBot

    from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
    import matplotlib.pyplot as plt 
    import cv2
    import tqdm

    tile_size=14
    optimal=False
    size='test'
    sizes = dict(
      test=dict(room_size=5, ntasks=1),
      small=dict(room_size=6, ntasks=1),
      medium=dict(room_size=9, ntasks=2),
      large=dict(room_size=12, ntasks=3),
      # xl=dict(room_size=10, ntasks=5),
      )
    task2reward={
          "slice" : 1,
          "chill" : 0,
          "clean" : 1,
          }
    if size == 'test':
      task2reward={"slice" : 1}
    elif size == 'small':
      task2reward={"slice" : 1, "chill" : 0}
    env = KitchenComboLevel(
        agent_view_size=5,
        task2reward=task2reward,
        use_time_limit=False,
        tile_size=tile_size,
        **sizes[size],
        )
    env = RGBImgPartialObsWrapper(env, tile_size=tile_size)

    def combine(full, partial):
        full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
        return np.concatenate((full_small, partial), axis=1)

    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)

    def move(action : str):
      # idx2action = {idx:action for action, idx in env.actions.items()}
      obs, reward, done, info = env.step(env.actions[action])
      full = env.render('rgb_array', tile_size=tile_size, highlight=True)
      window.show_img(combine(full, obs['image']))

    def show(obs):
      full = env.render('rgb_array', tile_size=tile_size, highlight=True)
      window.set_caption(obs['mission'])
      window.show_img(combine(full, obs['image']))
      # import ipdb; ipdb.set_trace()
      time.sleep(.05)

    for _ in tqdm.tqdm(range(100)):
      obs = env.reset()
      show(obs)
      if optimal:
        bot = GotoAvoidBot(env)
        obss, actions, rewards, dones = bot.generate_traj(
          plot_fn=show)
      else:
        rewards = []
        for step in range(1000):
            obs, reward, done, info = env.step(env.action_space.sample())
            rewards.append(reward)
            # print(reward, sum(rewards), done)
            # print('---------')
            show(obs)
            # if sum(rewards) > 0:
            #   import ipdb; ipdb.set_trace()
            if done:
              break

      print(f"Total reward: {sum(rewards)}")
      # print("Final occurrences:", env.object_occurrences)
      import ipdb; ipdb.set_trace()
