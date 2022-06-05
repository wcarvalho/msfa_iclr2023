
from gym import spaces

import math
import numpy as np
import copy
from envs.babyai_kitchen.world import Kitchen
from envs.babyai_kitchen.levelgen import KitchenLevel
import functools


MANUAL_TEST = True
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
                 walls_gone = False,
                 one_room = False,
                 mission_object = None,
                 deterministic_rooms = False,
                 room_reward = 0.0,
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
        self.one_room = one_room
        if one_room:
            walls_gone = False
            room_reward = 0
        self.deterministic_rooms = deterministic_rooms
        self.objectlist = objectlist
        self.walls_gone = walls_gone
        self.stop_when_gone = stop_when_gone
        self.doors_start_open = doors_start_open
        all_objects = set(functools.reduce(lambda x,y: x + list(y.keys()),objectlist,[]))
        self.num_objects = len(all_objects)
        self.pickup_required = pickup_required
        self.epsilon = epsilon
        self.verbosity = verbosity
        self.room = None #variable to keep track of what the primary room of the target object is
        self.room_location = None #same purpose, just instead of a number, it's 2 coords
        self.room_reward = room_reward
        self.got_room_reward = False
        self.carrying = None

        self.random_mission = True
        if mission_object:
            self.random_mission = False
            self.mission_object = mission_object

        #initialize the big objects we need
        kitchen = Kitchen(
            objects=all_objects,
            tile_size=tile_size,
            rootdir=rootdir,
            verbosity=verbosity,
        )
        #we need to reorder the type2idx dictionary based on the self.default_objects list
        self.default_objects = copy.deepcopy(kitchen.objects)
        self._task_objects = [o.name for o in self.default_objects]
        self.type2idx = {o: i for i, o in enumerate(self._task_objects)}

        #the mission array will just be one-hot over all the objects
        #stored in self.mission_arr
        self.select_mission()

        kwargs["task_kinds"] = ['goto','pickup']
        kwargs['actions'] = ['left', 'right', 'forward','pickup_contents', 'place_down']
        if (not one_room) and (not walls_gone):
            kwargs['actions'].append('open')
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
            shape=(self.num_objects + 4,),
            dtype='uint8'
        )
        self.observation_space.spaces['pickup'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.num_objects + 4,),
            dtype='uint8'
        )

    #resets self.mission_arr to a random mission (i.e. random object)
    def select_mission(self):
        self.mission_arr = np.zeros([self.num_objects],dtype=np.uint8)
        if self.random_mission:
            goal_idx = np.random.choice(range(self.num_objects))
        else:
            goal_idx = self.type2idx[self.mission_object]
        self.mission_arr[goal_idx] = 1
        if self.verbosity==1:
            print("Goal is to " + ("pickup " if self.pickup_required else "goto ") + self._task_objects[goal_idx])
            #print("mission: " + str(self.mission_arr))

    @property
    def task_objects(self):
        return self._task_objects

    def reset_task(self):
        self.select_mission()

        VALID_ROOMS_ = np.array([[0, 1], [1, 0], [2, 1]])
        DOOR_COLORS = np.array(['red', 'green', 'blue'])


        #we permute valid rooms and colors:
        if self.deterministic_rooms:
            perm = np.array([0,1,2])
        else:
            perm = np.random.permutation(3)
        VALID_ROOMS = VALID_ROOMS_[perm].tolist()
        # DOOR_COLORS = DOOR_COLORS[perm].tolist()

        # generate grid
        self._gen_grid(width=self.width, height=self.height)

        self.object_occurrences = np.zeros(self.num_objects, dtype=np.uint8)


        #place all of the objects
        for room_idx, room_objects in enumerate(self.objectlist):
            for (obj, num_to_place) in room_objects.items():
                placeable_obj = self.default_objects[self.type2idx[obj]]

                for _ in range(num_to_place):
                    #if one room just place all in the same room
                    if self.one_room:
                        self.place_in_room(1, 1, placeable_obj)
                        self.room = 0
                        self.room_location = (1,1)
                    else:
                        if self.mission_arr[self.type2idx[obj]]==1: #set self.room if this is indeed the reward object
                            self.room = perm[room_idx] + 1
                            self.room_location = tuple(VALID_ROOMS[room_idx])
                        #epsilon chance of random room placement
                        if np.random.uniform(0,1)<self.epsilon:
                            random_room = VALID_ROOMS[np.random.choice(range(len(VALID_ROOMS)))]
                            self.place_in_room(random_room[0], random_room[1], placeable_obj)
                        else:
                            self.place_in_room(VALID_ROOMS[room_idx][0], VALID_ROOMS[room_idx][1], placeable_obj)
                    if self.stop_when_gone:
                        self.object_occurrences[self.type2idx[obj]]+=int(self.mission_arr[self.type2idx[obj]]!=0)
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



        if self.walls_gone: #walls are never gone if one room
            self.remove_wall(1,1,0)
            self.remove_wall(1, 1, 2)
            self.remove_wall(1, 1, 3)
        else:
            if not self.one_room: #if one room, no doors
                door1, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[0])], DOOR_COLORS[0], locked=False)
                door2, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[1])], DOOR_COLORS[1], locked=False)
                door3, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[2])], DOOR_COLORS[2], locked=False)

                #potentially start with the doors open
                if self.doors_start_open:
                    door1.is_open = True
                    door2.is_open = True
                    door3.is_open = True

        #now we gotta update the mission arr based on the room reward
        room_embed_task = np.zeros(4,dtype=np.uint8)
        self.mission_arr = np.concatenate([self.mission_arr,room_embed_task],dtype=np.uint8)


    def reset(self):
        obs = super().reset()
        self.got_room_reward = False
        assert self.carrying is None
        obs['pickup'] = np.zeros(self.num_objects + 4, dtype=np.uint8) #plus 4 is for room we are in
        obs['pickup'][self.num_objects] = 1 #because we are in 0'th room
        obs['mission'] = self.mission_arr
        return obs

    def remove_object(self, fwd_pos, pickup_vector):
        # get reward
        object = self.grid.get(*fwd_pos)

        obj_type = object.type

        if obj_type in self._task_objects:
            obj_idx = self.type2idx[obj_type]

            action_info = self.kitchen.interact(
                action='pickup_contents',
                object_infront=object,
                fwd_pos=fwd_pos,
                grid=self.grid,
                env=self,  # only used for backwards compatibility with toggle
            )
            self.carrying = self.kitchen.carrying
            if self.verbosity==1:
                print(action_info)
            if action_info['success']:
                self.grid.set(*fwd_pos, None)
                pickup_vector[obj_idx] += 1
                reward = float(self.mission_arr[obj_idx])
                if self.remaining[obj_idx] > 0:
                    self.remaining[obj_idx] -= 1
            else:
                reward = 0.0



            return reward
        assert obj_type=="wall" or obj_type=="door"
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

        elif action == self.actiondict.get('place_down',-1):
            #first check that we are carrying something and that space in front is empty
            if self.carrying is not None and self.grid.get(*fwd_pos)==None:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying = None
                self.kitchen.update_carrying(None)


        # pickup or no-op if not pickup_required
        else:
            if object_infront and self.pickup_required and self.carrying is None:
                # get reward
                reward = self.remove_object(fwd_pos, pickup)


        #get the room reward for going in the right room for the first time
        # add room to pickup
        room_embed = np.zeros(4,dtype=np.uint8)
        ROOM_ORDER = [(0, 1), (1, 0), (2, 1)]
        if not self.got_room_reward:
            curr_i = self.agent_pos[0]//(self.room_size - 1)
            curr_j = self.agent_pos[1] // (self.room_size - 1)
            if (curr_i, curr_j)==self.room_location:
                reward+=self.room_reward
                self.got_room_reward = True
                if self.verbosity==1:
                    print("Got reward for entering correct room: {0}".format(self.room_reward))

                # check index of room we are in and add this to pickup
                if (curr_i, curr_j) in ROOM_ORDER:
                    room_embed[ROOM_ORDER.index((curr_i, curr_j)) + 1] = 1
                else:
                    room_embed[0] = 1

        pickup = np.concatenate([pickup, room_embed],dtype=np.uint8)

        # print("remaining: " + str(self.remaining))

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

        if not done:
            assert self.remaining.sum()>0

        obs = self.gen_obs()

        obs['mission'] = self.mission_arr
        obs['pickup'] = pickup

        if self.verbosity==1:
            print("Pickup: " + str(pickup))

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


    env = MultiroomGotoEnv(
        agent_view_size=5,
        objectlist=[{'pan': 1,'pot':1,'bowl':1}, {'tomato': 1,'lettuce':1, 'onion':1}, {'knife':1,'apple':1, 'orange':1}],
        pickup_required=True,
        tile_size=tile_size,
        epsilon=0.0,
        room_size=5,
        doors_start_open=True,
        stop_when_gone=True,
        walls_gone=False,
        verbosity=1,
        one_room=False,
        deterministic_rooms=False,
        room_reward=.25
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


    for _ in tqdm.tqdm(range(1000)):
        obs = env.reset()
        if MANUAL_TEST:
            full = env.render('rgb_array', tile_size=tile_size, highlight=True)
            window.set_caption(obs['mission'])
            window.show_img(combine(full, obs['image']))

        rewards = []
        # print("Initial occurrences:", env.object_occurrences)
        test_steps = 25
        if not MANUAL_TEST:
            test_steps = 100
        for step in range(test_steps):
            if MANUAL_TEST:
                input_action = int(eval(input()))
            else:
                input_action = env.action_space.sample()
            obs, reward, done, info = env.step(input_action)
            if MANUAL_TEST:
                print("reward: {0}".format(reward))
            rewards.append(reward)
            if MANUAL_TEST:
                full = env.render('rgb_array', tile_size=tile_size, highlight=True)
                window.show_img(combine(full, obs['image']))
            if done:
                break

        total_reward = sum(rewards)
        normalized_reward = total_reward / env.object_occurrences[0]
        # print("Final occurrences:", env.object_occurrences)
        print(f"Total reward: {total_reward}")
        # print(f"Normalized reward: {normalized_reward}")
        if MANUAL_TEST:
            import ipdb;

            ipdb.set_trace()

"""
Advice:
 - change 1 thing at a time, record clearly how it works YEP
 - r2d1_noise can make r2d1 noisier and get reward DONE
 - do prioritized replay DONE
 - maybe get rid of walls? --> make sure agent can see deep into rooms if you do this
 - Maybe do frequent updates?
 -

Q's:
    how do you evaluate on each task separately?
    error about ordering: why doesn't it break things?? How to fix?
    [actor/0] The table signature is:
[actor/0]     0: Tensor<name: 'observation/observation/image/0/observations/observation/image', dtype: uint8, shape: [?,50,50,3]>, 1: Tensor<name: 'observation/observation/pickup/0/observations/observation/pickup', dtype: uint8, shape: [?,3]>, 2: Tensor<name: 'observation/observation/task/0/observations/observation/task', dtype: uint8, shape: [?,3]>, 3: Tensor<name: 'observation/action/0/observations/action', dtype: int32, shape: [?]>, 4: Tensor<name: 'observation/reward/0/observations/reward', dtype: float, shape: [?]>, 5: Tensor<name: 'action/0/actions', dtype: int32, shape: [?]>, 6: Tensor<name: 'reward/0/rewards', dtype: float, shape: [?]>, 7: Tensor<name: 'discount/0/discounts', dtype: float, shape: [?]>, 8: Tensor<name: 'start_of_episode/start_of_episode', dtype: bool, shape: [?]>, 9: Tensor<name: 'extras/core_state/hidden/1/core_state/hidden', dtype: float, shape: [?,512]>, 10: Tensor<name: 'extras/core_state/cell/1/core_state/cell', dtype: float, shape: [?,512]>
[actor/0]
[actor/0] The provided trajectory signature is:
[actor/0]     0: Tensor<name: '0', dtype: uint8, shape: [31,3]>, 1: Tensor<name: '1', dtype: uint8, shape: [31,50,50,3]>, 2: Tensor<name: '2', dtype: uint8, shape: [31,3]>, 3: Tensor<name: '3', dtype: int32, shape: [31]>, 4: Tensor<name: '4', dtype: float, shape: [31]>, 5: Tensor<name: '5', dtype: int32, shape: [31]>, 6: Tensor<name: '6', dtype: float, shape: [31]>, 7: Tensor<name: '7', dtype: float, shape: [31]>, 8: Tensor<name: '8', dtype: bool, shape: [31]>, 9: Tensor<name: '9', dtype: float, shape: [31,512]>, 10: Tensor<name: '10', dtype: float, shape: [31,512]>.
[actor/0]

Ideas to make agent learn better:
Hyperparams:
 - bigger replay size, because reward is achieved so infrequently
 - change priority weight to get more samples with reward --> it's turned off rn
 - WILKA THINGS THIS WON'T MATTER: update the learner more frequently later on - variable_update period
 - bigger area the agent sees

    
 walkthrough of usfa train code
    look at msf nets.py
    w_train is all the w's
    by default during test time we GPI over all the w's
    by default task embed is identity
    ***In Jax you can't, just, uh, build stuff*** you gotta do it inside a ~Transform~ function
 
TODOs:
    make actual metrics for how well thing is learning
    make it learn
    evaluate on tasks separately to see how well it does each task
    look at vmap in losses usfa **this is confusing**
    
Learning Metrics:
Mean episode return for actor and evaluator after 10M epochs
Mean episode return per-task for actor and evaluator after 10M epochs
Same after 2M epochs
Loss of learner... but this one doesn't seem to be a problem
Once colocation is introduced:
    correlation coefs between colocated object returns versus non-colocated returns
    
    
Bug Fixes:
 - run a bunch of times to make sure it's not random (both distributed and single) DONE
 - rework nets file (int casting is suspicious) (also check out BasicRecurrent class) DONE FOR R2D1
 - print out specs all over the place NO NEED
 - add a bunch of assertions in environment and sample many episodes with those in there DONE
 - potentially simplify to one room
 
  - transfer ideas to skill discovery
"""
