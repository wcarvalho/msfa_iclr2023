"""

"""

from gym import spaces

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

        self.mission_arr = np.array(
            list(self.object2reward.values()),
            dtype=np.uint8,
            )
        self.object_names = list(object2reward.keys())
        if objects:
          assert objects == self.object_names
        else:
          objects = self.object_names

        self.object2idx = {o:idx for idx, o in enumerate(objects)}
        self._task_oidxs = [self.object2idx[o] for o in self._task_objects]
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
            dtype='uint8'
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
            object_idx = self.object2idx[object_type]
            self.object_occurrences[object_idx] += 1
            object = self.default_objects[object_idx]
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

    def step(self, action):
        """Copied from: 
        - gym_minigrid.minigrid:MiniGridEnv.step
        - babyai.levels.levelgen:RoomGridLevel.step
        This class derives from RoomGridLevel. We want to use the parent of RoomGridLevel for step. 
        """
        # ======================================================
        # copied from MiniGridEnv
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
        action_info = None
        if action == self.actiondict.get('left', -1):
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
        # pickup
        else:
            if object_infront:
                # get reward
                if object_infront.type in self.object2reward:
                    obj_type = object_infront.type
                    obj_idx = self.object2idx[obj_type]

                    reward = float(self.object2reward[obj_type])
                    self.grid.set(*fwd_pos, None)

                    if self.respawn:
                      # move object
                      self.place_in_room(0, 0, object_infront)
                      self.object_occurrences[obj_idx] += 1
                    else:
                      self.remaining[obj_idx] -= 1

                    pickup[obj_idx] = 1

        # ======================================================
        # copied from RoomGridLevel
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
    size='small'
    sizes = dict(
      small=dict(room_size=5, nobjects=1),
      medium=dict(room_size=8, nobjects=2),
      large=dict(room_size=10, nobjects=3),
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

    for _ in tqdm.tqdm(range(1000)):
      obs = env.reset()
      full = env.render('rgb_array', tile_size=tile_size, highlight=True)
      window.set_caption(obs['mission'])
      window.show_img(combine(full, obs['image']))

      rewards = []
      # print("Initial occurrences:", env.object_occurrences)
      for step in range(5):
          obs, reward, done, info = env.step(env.action_space.sample())
          rewards.append(reward)
          full = env.render('rgb_array', tile_size=tile_size, highlight=True)
          window.show_img(combine(full, obs['image']))
          if done:
            break

      total_reward = sum(rewards)
      normalized_reward = total_reward/env.object_occurrences[0]
      # print("Final occurrences:", env.object_occurrences)
      # print(f"Total reward: {total_reward}")
      # print(f"Normalized reward: {normalized_reward}")
    import ipdb; ipdb.set_trace()
