
from gym import spaces

import math
import numpy as np
import copy
from envs.babyai_kitchen.world import Kitchen
from envs.babyai_kitchen.levelgen import KitchenLevel
import functools


MANUAL_TEST = False
"""
Assumptions:
6 rooms, agent always starts in bottom center room
2 rooms empty, only 3 have stuff
doors always have same colors, empty rooms are locked
_____________________       
|empty|       |empty|
|_____|_______|_____|
|     | start |     |
|_____|_______|_____|

"""
class MultiroomGotoEnv(KitchenLevel):
    def __init__(self,
                 *args,
                 objectlist,
                 tile_size=8,
                 rootdir='.',
                 verbosity=0,
                 pickup_required=True,
                 epsilon = 0.0,
                 room_size = 8,
                 doors_start_open = False,
                 stop_when_gone = False,
                 **kwargs):
        """Summary

        Args:
            *args: Description
            objectlist (TYPE): A nested list of [{object_name: object_quantity}] one dictionary for each of the three rooms
            tile_size (int, optional): how many pixels to use for a tile, I think
            rootdir (str, optional): Just a path for the kitchen env to search for files, probably leave this be
            verbosity (int, optional): how much to print

            epsilon (float, optional): chance that an object is not in its usual room
            room_size (int, optional): the size of a room, duh
            doors_start_open (bool, optional): make the doors start open (default is closed but unlocked)
            stop_when_gone (bool, optional): should we stop the episode when all the objects with reward associated are gone?
            **kwargs: Description
        """

        #assert args are legit
        assert len(objectlist)==3

        #define all the objects in our world
        #any object can be an objective in this environment, so we don't need to keep track of which are pick-up-able
        self.objectlist = objectlist
        self.stop_when_gone = stop_when_gone
        self.doors_start_open = doors_start_open
        self._task_objects = functools.reduce(lambda x,y: x + list(y.keys()),objectlist,[]) #Rename this
        self.num_objects = len(self._task_objects)
        self.pickup_required = pickup_required
        self.epsilon = epsilon
        self.verbosity = verbosity

        #the mission array will just be one-hot over all the objects
        #stored in self.mission_arr
        self.select_mission()

        #initialize the big objects we need
        kitchen = Kitchen(
            objects=self._task_objects,
            tile_size=tile_size,
            rootdir=rootdir,
            verbosity=verbosity,
        )
        #we need to reorder the type2idx dictionary based on the self.default_objects list
        self.default_objects = copy.deepcopy(kitchen.objects)
        self._task_objects = [o.name for o in self.default_objects]
        self.type2idx = {o: i for i, o in enumerate(self._task_objects)}

        kwargs["task_kinds"] = ['goto','pickup']
        kwargs['actions'] = ['left', 'right', 'forward', 'open','pickup_contents']
        kwargs['kitchen'] = kitchen
        super().__init__(
            *args,
            tile_size=tile_size,
            rootdir=rootdir,
            verbosity=verbosity,
            objects=self._task_objects,
            num_rows=2,
            num_cols=3,
            room_size=room_size,
            **kwargs)

        self.observation_space.spaces['mission'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.num_objects,),
            dtype='uint8'
        )
        self.observation_space.spaces['pickup'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.num_objects,),
            dtype='uint8'
        )

    #resets self.mission_arr to a random mission (i.e. random object)
    def select_mission(self):
        self.mission_arr = np.zeros([self.num_objects],dtype=np.uint8)
        goal_idx = np.random.choice(range(self.num_objects))
        self.mission_arr[goal_idx] = 1
        if self.verbosity==1:
            print("Goal is to " + ("pickup " if self.pickup_required else "goto ") + self._task_objects[goal_idx])

    @property
    def task_objects(self):
        return self._task_objects

    def reset_task(self):

        VALID_ROOMS = np.array([[0, 1], [1, 0], [2, 1]])
        DOOR_COLORS = np.array(['red', 'green', 'blue'])


        #we permute valid rooms and colors:
        perm = np.random.permutation(3)
        VALID_ROOMS = VALID_ROOMS[perm].tolist()
        # DOOR_COLORS = DOOR_COLORS[perm].tolist()

        # generate grid
        self._gen_grid(width=self.width, height=self.height)

        self.object_occurrences = np.zeros(self.num_objects, dtype=np.uint8)


        #place all of the objects
        for room_idx, room_objects in enumerate(self.objectlist):
            for obj_idx,(obj, num_to_place) in enumerate(room_objects.items()):
                placeable_obj = self.default_objects[self.type2idx[obj]]

                for _ in range(num_to_place):
                    #epsilon chance of random room placement
                    if np.random.uniform(0,1)<self.epsilon:
                        random_room = VALID_ROOMS[np.random.choice(range(len(VALID_ROOMS)))]
                        self.place_in_room(random_room[0], random_room[1], placeable_obj)
                    else:
                        self.place_in_room(VALID_ROOMS[room_idx][0], VALID_ROOMS[room_idx][1], placeable_obj)
                    if self.stop_when_gone:
                        self.object_occurrences[self.type2idx[obj]]+=self.mission_arr[self.type2idx[obj]]
                    else:
                        self.object_occurrences[self.type2idx[obj]]+=1

        self.remaining = np.array(self.object_occurrences)
        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent(1,1)
            start_room = self.room_from_pos(*self.agent_pos)
            break


        # arrange the door colors and locks
        # Order of the doors is right, down, left, up
        #we have 3 doors we care about
        #create the 3 doors to the starting room
        room_to_door = {(0,1):2,(1,0):3,(2,1):0}

        door1, _ = self.add_door(1,1,room_to_door[tuple(VALID_ROOMS[0])],DOOR_COLORS[0],locked=False)
        door2, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[1])],DOOR_COLORS[1],locked=False)
        door3, _ =self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[2])],DOOR_COLORS[2],locked=False)

        #potentially start with the doors open
        if self.doors_start_open:
            door1.is_open = True
            door2.is_open = True
            door3.is_open = True


    def reset(self):
        obs = super().reset()
        self.select_mission()
        assert self.carrying is None
        obs['pickup'] = np.zeros(self.num_objects, dtype=np.uint8)
        obs['mission'] = self.mission_arr

        return obs

    def remove_object(self, fwd_pos, pickup_vector):
        # get reward
        object = self.grid.get(*fwd_pos)

        obj_type = object.type

        if obj_type in self._task_objects:
            obj_idx = self.type2idx[obj_type]

            pickup_vector[obj_idx] = 1

            reward = float(self.mission_arr[obj_idx])
            self.grid.set(*fwd_pos, None)

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
        # copied from MiniGridEnv
        # ======================================================
        self.step_count += 1

        reward = 0.0
        pickup = np.zeros(self.num_objects, dtype=np.uint8)
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

            if object_infront and not self.pickup_required:
                reward = self.remove_object(fwd_pos, pickup)

        elif action == self.actiondict.get('open',-1):
            if object_infront is not None:
                if object_infront.type=='door':
                    object_infront.is_open = not object_infront.is_open

        # pickup or no-op if not pickup_required
        else:
            if object_infront and self.pickup_required:
                # get reward
                reward = self.remove_object(fwd_pos, pickup)

        # ======================================================
        # copied from RoomGridLevel
        # ======================================================
        info = {}
        # if past step count, done
        if self.step_count >= self.max_steps and self.use_time_limit:
            done = True

        # if no task objects remaining, done
        remaining = self.remaining.sum()
        if remaining < 1e-5:
            done = True

        obs = self.gen_obs()

        obs['mission'] = self.mission_arr
        obs['pickup'] = pickup

        return obs, reward, done, info


if __name__ == '__main__':
    import gym_minigrid.window
    import time
    from rljax.envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
    import matplotlib.pyplot as plt
    import cv2
    import tqdm
    import os
    from babyai.levels.iclr19_levels import Level_GoToImpUnlock

    #os.chdir('../..')

    tile_size = 10

    # env = MultiroomGoto(
    #     agent_view_size=5,
    #     objectlist=[{'pan':4,'pot':4,'stove':1}, {'tomato':1,'potato':10}, {'orange':5,'apple':2}],
    #     pickup_required=False,
    #     tile_size=tile_size,
    #     epsilon = .1,
    #     room_size=10,
    #     doors_start_open=False,
    #     stop_when_gone=True
    # )
    env = MultiroomGotoEnv(
        agent_view_size=5,
        objectlist=[{'pan': 1}, {'tomato': 1}, {'knife':1}],
        pickup_required=False,
        tile_size=tile_size,
        epsilon=0.0,
        room_size=5,
        doors_start_open=True,
        stop_when_gone=True
    )

    #env = Level_GoToImpUnlock(num_rows=2,num_cols=3)

    env = RGBImgPartialObsWrapper(env, tile_size=tile_size)


    def combine(full, partial):
        full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
        return np.concatenate((full_small, partial), axis=1)


    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)


    def move(action: str):
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
            if MANUAL_TEST:
                input_action = int(eval(input()))
            else:
                input_action = env.action_space.sample()
            obs, reward, done, info = env.step(input_action)
            print("reward: {0}".format(reward))
            rewards.append(reward)
            full = env.render('rgb_array', tile_size=tile_size, highlight=True)
            window.show_img(combine(full, obs['image']))
            if done:
                break

        total_reward = sum(rewards)
        normalized_reward = total_reward / env.object_occurrences[0]
        # print("Final occurrences:", env.object_occurrences)
        print(f"Total reward: {total_reward}")
        # print(f"Normalized reward: {normalized_reward}")
        import ipdb;

        ipdb.set_trace()

"""NOTES:
evaluate on tasks separately to see how well it does each task
look at vmap in losses usfa **this is confusing**
when you read papers highlighting problems can be a handy thing (and share them with Wilka)

 Sanity check has none colocated, just single object in each room (or just one room is fine)
 Start with just 2 rooms other than start room, 2-3 objects per room to compare new algo with existing
 
 Next round Q's:
    how does logging work? Can I use WANDB?
    
     
     walkthrough of usfa train code
        look at msf nets.py
        w_train is all the w's
        by default during test time we GPI over all the w's
        by default task embed is identity
        ***In Jax you can't, just, uh, build stuff*** you gotta do it inside a ~Transform~ function
        
use distributed train
multiprocessing and not multithreading
burn-in 0
episode length tuning
"""