"""

"""

from gym import spaces

import math
import numpy as np
import copy
from envs.babyai_kitchen.world import Kitchen
from envs.babyai_kitchen.levelgen import KitchenLevel

class GotoAvoidEnv(KitchenLevel):
  """docstring for GotoAvoidEnv"""
  def __init__(self, 
    *args,
    object2reward,
    tile_size=8,
    rootdir='.',
    verbosity=0,
    nobjects=2,
    respawn=False,
    kitchen=None,
    objects=None,
    pickup_required=True,
    **kwargs):
    """Summary
    
    Args:
        *args: Description
        object2reward (TYPE): Description
        tile_size (int, optional): Description
        rootdir (str, optional): Description
        verbosity (int, optional): Description
        nobjects (int, optional): How many times to spawn each objects type
        respawn (bool, optional): Description
        kitchen (None, optional): Description
        **kwargs: Description
    """
    self.object2reward = object2reward
    self._task_objects = [k for k,v in object2reward.items() if v > 0]
    self.pickup_required = pickup_required

    self.mission_arr = np.array(
        list(self.object2reward.values()),
        dtype=np.int32,
        )
    self.object_names = list(object2reward.keys())
    if objects:
      assert objects == self.object_names
    else:
      objects = self.object_names

    self.type2idx = {o:idx for idx, o in enumerate(objects)}
    self._task_oidxs = [self.type2idx[o] for o in self._task_objects]
    self.nobjects = nobjects
    self.respawn = respawn
    kitchen = kitchen or Kitchen(
        objects=objects,
        tile_size=tile_size,
        rootdir=rootdir,
        verbosity=verbosity,
        )
    self.default_objects = copy.deepcopy(kitchen.objects)

    kwargs["task_kinds"] = ['pickup']
    kwargs['actions'] = ['left', 'right', 'forward', 'pickup_contents']
    kwargs['kitchen'] = kitchen
    super().__init__(
        *args,
        tile_size=tile_size,
        rootdir=rootdir,
        verbosity=verbosity,
        objects=objects,
        **kwargs)

    self.observation_space.spaces['mission'] = spaces.Box(
        low=0,
        high=255,
        shape=(len(self.object2reward),),
        dtype='int32'
    )
    self.observation_space.spaces['pickup'] = spaces.Box(
        low=0,
        high=255,
        shape=(len(self.object2reward),),
        dtype='uint8'
    )

  @property
  def task_objects(self):
    return self._task_objects

  def reset_task(self):
    # generate grid
    self._gen_grid(width=self.width, height=self.height)

    # connect all rooms
    self.connect_all()

    # -----------------------
    # get objects to spawn
    # -----------------------
    types_to_place = []
    for object_type in self.object2reward.keys():
      types_to_place.extend([object_type]*self.nobjects)

    # -----------------------
    # spawn objects
    # -----------------------
    self.object_occurrences = np.zeros(len(self.object2reward), dtype=np.uint8)
    for object_type in types_to_place:
        object_idx = self.type2idx[object_type]
        self.object_occurrences[object_idx] += 1
        object = self.default_objects[object_idx]
        assert object.type in self.object2reward, "placing object with no reward signature"
        self.place_in_room(0, 0, object)

    self.remaining = np.array(self.object_occurrences)
    # The agent must be placed after all the object to respect constraints
    while True:
        self.place_agent()
        start_room = self.room_from_pos(*self.agent_pos)
        # Ensure that we are not placing the agent in the locked room
        if start_room is self.locked_room:
            continue
        break

  def reset(self):
    obs = super().reset()
    assert self.carrying is None
    obs['pickup'] = np.zeros(len(self.object2reward), dtype=np.uint8)
    obs['mission'] = self.mission_arr


    return obs

  def remove_object(self, fwd_pos, pickup_vector):
    # get reward
    object = self.grid.get(*fwd_pos)

    if object.type in self.object2reward:
      obj_type = object.type
      obj_idx = self.type2idx[obj_type]
      pickup_vector[obj_idx] = 1

      reward = float(self.object2reward[obj_type])
      self.grid.set(*fwd_pos, None)

      if self.respawn:
        # move object
        self.place_in_room(0, 0, object)
        self.object_occurrences[obj_idx] += 1
      else:
        self.remaining[obj_idx] -= 1

      return reward
    return 0.0

  def place_obj(self,
    obj,
    top=None,
    size=None,
    reject_fn=None,
    max_tries=math.inf
   ):
    """
    Place an object at an empty position in the grid

    :param top: top-left position of the rectangle where to place
    :param size: size of the rectangle where to place
    :param reject_fn: function to filter out potential positions
    """

    if top is None:
        top = (0, 0)
    else:
        top = (max(top[0], 0), max(top[1], 0))

    if size is None:
        size = (self.grid.width, self.grid.height)

    num_tries = 0

    while True:
        # This is to handle with rare cases where rejection sampling
        # gets stuck in an infinite loop
        if num_tries > max_tries:
            raise RecursionError('rejection sampling failed in place_obj')

        num_tries += 1

        pos = np.array((
            self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
            self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
        ))

        # Don't place the object on top of another object
        if self.grid.get(*pos) != None:
            continue

        # Don't place the object where the agent is
        if np.array_equal(pos, self.agent_pos):
            continue


        # # Check if there is a filtering criterion
        # if reject_fn and reject_fn(self, pos):
        #     continue

        break

    self.grid.set(*pos, obj)

    if obj is not None:
        obj.init_pos = pos
        obj.cur_pos = pos

    return pos

  def step(self, action):
    """Copied from: 
    - gym_minigrid.minigrid:MiniGridEnv.step
    - babyai.levels.levelgen:RoomGridLevel.step
    This class derives from RoomGridLevel. We want to use the parent of RoomGridLevel for step. 
    """
    # ======================================================
    # adapted from MiniGridEnv
    # ======================================================
    self.step_count += 1

    reward = 0.0
    pickup = np.zeros(len(self.object2reward), dtype=np.uint8)
    done = False

    # Get the position in front of the agent
    fwd_pos = self.front_pos

    # Get the contents of the cell in front of the agent
    object_infront = self.grid.get(*fwd_pos)

    # Rotate left
    if action == self.actiondict.get('noop', -1):
      pass
    elif action == self.actiondict.get('left', -1):
        self.agent_dir -= 1
        if self.agent_dir < 0:
            self.agent_dir += 4

    # Rotate right
    elif action == self.actiondict.get('right', -1):
        self.agent_dir = (self.agent_dir + 1) % 4

    # Move forward
    elif action == self.actiondict.get('forward', -1):
        if object_infront == None or object_infront.can_overlap():
            self.agent_pos = fwd_pos


        if object_infront and not self.pickup_required:
          reward = self.remove_object(fwd_pos, pickup)
    # pickup or no-op if not pickup_required
    elif action == self.actiondict.get('pickup_contents', -1):
        if object_infront and self.pickup_required:
          # get reward
          reward = self.remove_object(fwd_pos, pickup)
    else:
      raise RuntimeError(action)

    # ======================================================
    # adapted from RoomGridLevel
    # ======================================================
    info = {}
    # if past step count, done
    if self.step_count >= self.max_steps and self.use_time_limit:
        done = True

    # if no task objects remaining, done
    remaining = self.remaining[self._task_oidxs].sum()
    if remaining < 1e-5:
      done = True

    obs = self.gen_obs()

    obs['mission'] = self.mission_arr
    obs['pickup'] = pickup

    return obs, reward, done, info

if __name__ == '__main__':
    import gym_minigrid.window
    import time
    from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
    import matplotlib.pyplot as plt 
    import cv2
    import tqdm

    tile_size=20
    size='large'
    sizes = dict(
      small=dict(room_size=5, nobjects=1),
      medium=dict(room_size=7, nobjects=2),
      large=dict(room_size=8, nobjects=3),
      xl=dict(room_size=10, nobjects=5),
      )

    env = GotoAvoidEnv(
        agent_view_size=5,
        object2reward={
            "pan" : 1,
            "plates" : 0,
            "tomato" : 0,
            "knife" : 0,
            },
        respawn=False,
        pickup_required=True,
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

    for _ in tqdm.tqdm(range(3)):
      obs = env.reset()
      full = env.render('rgb_array', tile_size=tile_size, highlight=True)
      window.set_caption(obs['mission'])
      window.show_img(combine(full, obs['image']))

      rewards = []
      # print("Initial occurrences:", env.object_occurrences)
      for step in range(25):
          obs, reward, done, info = env.step(env.action_space.sample())
          rewards.append(reward)
          full = env.render('rgb_array', tile_size=tile_size, highlight=True)
          window.show_img(combine(full, obs['image']))
          if done:
            break

      total_reward = sum(rewards)
      normalized_reward = total_reward/env.object_occurrences[0]
      # print("Final occurrences:", env.object_occurrences)
      print(f"Total reward: {total_reward}")
      # print(f"Normalized reward: {normalized_reward}")
      import ipdb; ipdb.set_trace()
