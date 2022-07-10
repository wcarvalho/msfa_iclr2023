
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
door color corresponds to which objects are in each room
We number the rooms 0, 1, 2, 3
_____________________
|empty|   2   |empty|
|_____|_______|_____|
|  1  | start |  3  |
|_____|___0___|_____|

You can just run this file with MANUAL_TEST set to True to manually walk around the environment
Just give integer inputs to stdin to control the agent.

The descriptions of each argument to the environment below should help give a decent sense of how the Env works

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
                 room_size = 5,
                 doors_start_open = False,
                 stop_when_gone = False,
                 walls_gone = False,
                 one_room = False,
                 two_rooms = False,
                 mission_object = None,
                 deterministic_rooms = False,
                 room_reward = 0.25,
                 room_reward_task_vector = True,
                 **kwargs):
        """Summary

        Args:
            *args: Description
            objectlist (TYPE): A nested list of [{object_name: object_quantity}] one dictionary for each of the three rooms
            tile_size (int, optional): how many pixels to use for a tile
            rootdir (str, optional): Just a path for the kitchen env to search for files, probably leave this be
            verbosity (int, optional): set this to 1 if you want a bunch of output to stdout
            pickup_required (bool, optional): This should generally be True. It determines whether the task is to just goto
                each object, or actually pick it up
            epsilon (float, optional): chance that an object is not in its usual room
            room_size (int, optional): the size of a room in grid units. Minimum (and recommended value) is 5
            doors_start_open (bool, optional): make the doors start open (default is closed but unlocked)
            stop_when_gone (bool, optional): If this is true, the episode ends when the objects with positive associated
                reward are gone. Otherwise, the episode only ends if every object is picked up.
            walls_gone (bool, optional): If you set this to true, there will be no walls between the rooms with objects
            one_room (bool, optional): If you set this to True, We put walls without doors around the starting room.
                All objects will be placed in the starting room in this case.
            mission_object (string, Optional) This exists for use with the acme wrapper. It allows you to pass an object
                to the environment which is always set as the target object every single episode. If it is None,
                a random target will be chosen, which is standard behavior for debugging. However, mission_object is used
                when working with the acme wrapper because it allows you to assign a single object to a "level"
            deterministic_rooms (bool, Optional) If set to true, this argument makes it so that the objects in each room
                are the same between episodes. When this is false (standard behavior) is to permute the room order so that the same objects
                remain colocated and correspond to the same door color, but the room they are jointly in varies.
            room_reward (float, Optional) You can give a small bonus reward to the agent for entering the room of the correct object.
                This often seems to be necessary for learning. The reward is given at most once per episode
            room_reward_task_vector (bool, Optional) If this set to True, the task vector and pickup vector will be augmented
                by 4 dimensions. The added dimensions of the pickup vector will keep track of the room the first time
                the agent enters each room, flipping the corresponding index to 1 at the moment of entry
            **kwargs: Description
        """

        #Even if 2 rooms, we just leave things as they are cut cut off the top room
        assert len(objectlist) == 3
        self.number_of_rooms = 3
        #define all the objects in our world
        #any object can be an objective in this environment, so we don't need to keep track of which are pick-up-able
        #be sure to give input objects which are allowed to be picked up!!
        self.one_room = one_room
        if one_room:
            walls_gone = False
            two_rooms = False
            room_reward = 0
            self.number_of_rooms = 1
        self.two_rooms = two_rooms


        self.deterministic_rooms = deterministic_rooms
        self.walls_gone = walls_gone
        self.stop_when_gone = stop_when_gone
        self.doors_start_open = doors_start_open

        if self.two_rooms:
            all_objects = set(functools.reduce(lambda x, y: x + list(y.keys()), objectlist[:2],
                                               []))  # a set of every object in our environment, ignore the last room
            self.objectlist = objectlist[:2]
            self.number_of_rooms = 2
        else:
            all_objects = set(functools.reduce(lambda x,y: x + list(y.keys()),objectlist,[])) #a set of every object in our environment
            self.objectlist = objectlist
        self.num_objects = len(all_objects) #the number of unique objects in our environment
        self.pickup_required = pickup_required
        self.epsilon = epsilon
        self.verbosity = verbosity
        self.room = None #variable to keep track of what the room of the target object is (RESET EVERY EPISODE)
        self.room_location = None #same as above, just instead of a single index, it's 2 coords (RESET EVERY EPISODE)
        self.room_reward = room_reward
        self.got_room_reward = False #variable to keep track of if we have received a room reward yet in each episode (RESET EVERY EPISODE)
        self.carrying = None #variable to keep track of what object we are carrying (CHANGES EACH STEP)
        self.visited_rooms = [] #keeps track of which rooms we have visited during the episode (RESET EVERY EPISODE)

        #if we have specified a mission object, we choose that as our mission every time
        self.random_mission = True
        if mission_object:
            self.random_mission = False
            self.mission_object = mission_object

        #initialize the kitchen object
        kitchen = Kitchen(
            objects=all_objects,
            tile_size=tile_size,
            rootdir=rootdir,
            verbosity=verbosity,
        )
        #we need to order the type2idx dictionary based on the self.default_objects list
        self.default_objects = copy.deepcopy(kitchen.objects)
        self._task_objects = [o.name for o in self.default_objects]
        self.type2idx = {o: i for i, o in enumerate(self._task_objects)}

        # 2 options: include room reward in task vector or not
        # If we include room reward in task vector, we update our train_tasks variable accordingly
        # Otherwise self.train_tasks is just identity
        # Be sure to use the environment's train_tasks variable for USFA/MSF for this reason and don't just \
        # always set train_tasks to identity
        self.room_reward_task_vector = room_reward_task_vector
        if room_reward_task_vector:
            self.task_vector_dims = self.num_objects + 4
            train_tasks = []
            for room_idx, roomdict in enumerate(self.objectlist):
                for obj, _ in roomdict.items():
                    task_vector = np.zeros(self.num_objects + 4)
                    obj_idx = self.type2idx[obj]
                    task_vector[obj_idx] = 1
                    task_vector[self.num_objects + room_idx + 1] = self.room_reward
                    train_tasks.append(task_vector)
            self.train_tasks = np.stack(train_tasks)
        else:
            self.task_vector_dims = self.num_objects
            self.train_tasks = np.eye(self.num_objects)


        #define the action space
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

        self.observation_space.spaces['mission'] = spaces.Box( #note we need floats to encode room reward
            low=0,
            high=255,
            shape=(self.task_vector_dims,),
            dtype='float32'
        )
        self.observation_space.spaces['pickup'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.task_vector_dims,),
            dtype='uint8'
        )

        self.observation_space.spaces['train_tasks'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.num_objects,self.task_vector_dims,),
            dtype='float32'
        )

    # Resets self.mission_arr to a random mission (i.e. random object) unless self.mission_object is defined
    # NOTE: this function does not deal with the extra dimensions of the mission corresponding to room reward.
    # That is dealt with in the reset_task function.
    def select_mission(self):
        self.mission_arr = np.zeros([self.num_objects],dtype=np.float32)

        if self.random_mission:
            goal_idx = np.random.choice(range(self.num_objects))
        else:
            goal_idx = self.type2idx[self.mission_object]

        self.mission_arr[goal_idx] = 1
        if self.verbosity==1:
            print("Goal is to " + ("pickup " if self.pickup_required else "goto ") + self._task_objects[goal_idx])

    @property
    def task_objects(self):
        return self._task_objects

    def reset_task(self):
        self.select_mission() #first we select which object we are going to be picking up

        # These lists contain the rooms where objects will be placed (if one_room isn't True) and the door colors
        # The door colors DO NOT necessarily correspond to the rooms above them.
        # Recall that colors correspond to the objects in the room, not the room's location
        if self.two_rooms:
            VALID_ROOMS_ = np.array([[0, 1], [1, 0]])
            DOOR_COLORS = np.array(['red', 'green'])
        else:
            VALID_ROOMS_ = np.array([[0, 1], [1, 0], [2, 1]])
            DOOR_COLORS = np.array(['red', 'green', 'blue'])


        # we permute valid rooms to randomize room order when deterministic_rooms is False
        if self.deterministic_rooms:
            perm = np.arange(self.number_of_rooms)
        else:
            perm = np.random.permutation(self.number_of_rooms)
        VALID_ROOMS = VALID_ROOMS_[perm].tolist()

        # generate grid
        self._gen_grid(width=self.width, height=self.height)

        #initialize the object_occurrences array to hold which objects are in the environment
        self.object_occurrences = np.zeros(self.num_objects, dtype=np.uint8)


        # place all of the objects
        for room_idx, room_objects in enumerate(self.objectlist): # For each room ...
            for (obj, num_to_place) in room_objects.items(): # For each object in each room ...
                placeable_obj = self.default_objects[self.type2idx[obj]] #get the object

                for _ in range(num_to_place): # we can place multiple of each object...

                    #if one_room is True just place all objects in the starting room
                    if self.one_room:
                        self.place_in_room(1, 1, placeable_obj)
                        self.room = 0
                        self.room_location = (1,1)

                    else:
                        if self.mission_arr[self.type2idx[obj]]==1: #set self.room if this is indeed the reward object
                            self.room = perm[room_idx] + 1
                            self.room_location = tuple(VALID_ROOMS[room_idx])

                        # epsilon chance of random room placement
                        if np.random.uniform(0,1)<self.epsilon:
                            random_room = VALID_ROOMS[np.random.choice(range(len(VALID_ROOMS)))]
                            self.place_in_room(random_room[0], random_room[1], placeable_obj)

                        else:
                            self.place_in_room(VALID_ROOMS[room_idx][0], VALID_ROOMS[room_idx][1], placeable_obj)

                    # object_occurrences only keeps track of objects relevant to determining if the episode is over
                    # If stop_when_gone is True, we only keep track of objects for which there is nonzero reward
                    if self.stop_when_gone:
                        self.object_occurrences[self.type2idx[obj]]+=int(self.mission_arr[self.type2idx[obj]]!=0)
                    else:
                        self.object_occurrences[self.type2idx[obj]]+=1

        # remaining is like object_occurrences, but updates as the episode progresses
        self.remaining = np.array(self.object_occurrences)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent(1,1)
            start_room = self.room_from_pos(*self.agent_pos)
            break

        # arrange the door colors and locks
        # Order of the doors in a given room is right, down, left, up
        # we have 3 doors we care about
        # create the 3 doors to the starting room
        room_to_door = {(0,1):2,(1,0):3,(2,1):0}

        # walls are never gone if one room
        # If walls are gone we get rid of walls and don't worry about doors
        if self.walls_gone:
            self.remove_wall(1,1,0)
            self.remove_wall(1, 1, 2)
            self.remove_wall(1, 1, 3)
        else:
            if not self.one_room: # if we have multiple rooms, we add doors to the starting room (unlocked)
                door1, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[0])], DOOR_COLORS[0], locked=False)
                door2, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[1])], DOOR_COLORS[1], locked=False)
                if not self.two_rooms:
                    door3, _ = self.add_door(1, 1, room_to_door[tuple(VALID_ROOMS[2])], DOOR_COLORS[2], locked=False)

                #potentially start with the doors open
                if self.doors_start_open:
                    door1.is_open = True
                    door2.is_open = True
                    if not self.two_rooms:
                        door3.is_open = True

        #now we have to update the mission arr based on the room reward
        if self.room_reward_task_vector:
            room_embed_task = np.zeros(4,dtype=np.float32)
            room_embed_task[self.room] = self.room_reward
            self.mission_arr = np.concatenate([self.mission_arr,room_embed_task],dtype=np.float32)

        if self.verbosity==1:
            print("Mission array: " + str(self.mission_arr))

    def reset(self):
        obs = super().reset() # this calls reset_task

        # variables that need to be reset each episode that aren't handled by reset_task
        self.got_room_reward = False
        self.visited_rooms = [(1,1)] #we start by visiting the 0'th room
        assert self.carrying is None

        #define the starting observation
        if self.room_reward_task_vector:
            obs['pickup'] = np.zeros(self.num_objects + 4, dtype=np.uint8) #plus 4 is for room we are in
            obs['pickup'][self.num_objects] = 1 # because we are in 0'th room for the first time
        else:
            obs['pickup'] = np.zeros(self.num_objects,dtype=np.uint8)
        obs['mission'] = self.mission_arr
        obs['train_tasks'] = self.train_tasks #this does not change episode-to-episode
        return obs

    def remove_object(self, fwd_pos, pickup_vector):
        # get the object that was picked up or gone-to
        object = self.grid.get(*fwd_pos)
        obj_type = object.type

        # This should always be true since all of our objects are potential task objects
        if obj_type in self._task_objects:
            obj_idx = self.type2idx[obj_type]

            #we need to figure out if we can pickup the object, and if so, pick it up
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

            #if we actually pick up the object, remove it from in front of the agent and decrease "remaining"
            #also assign reward
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

    #This function is just copied from goto_avoid
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
        """Based off (but not totally copied from):
        - gym_minigrid.minigrid:MiniGridEnv.step
        - babyai.levels.levelgen:RoomGridLevel.step
        This class derives from RoomGridLevel. We want to use the parent of RoomGridLevel for step.
        """

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

        #place-down action: If we are carrying something, set it down in front of the agent
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


        # get the room reward for going in the right room for the first time
        # add room to pickup if room_reward_task_vector is True
        room_embed = np.zeros(4,dtype=np.uint8)
        ROOM_ORDER = [(1,1), (0, 1), (1, 0), (2, 1)]
        curr_i = self.agent_pos[0] // (self.room_size - 1) # location of current room
        curr_j = self.agent_pos[1] // (self.room_size - 1)
        if not self.got_room_reward: # if we haven't already received room reward, we could get it again
            if (curr_i, curr_j)==self.room_location:
                reward+=self.room_reward
                self.got_room_reward = True
                if self.verbosity==1:
                    print("Got reward for entering correct room: {0}".format(self.room_reward))


        # check index of room we are in and add this to pickup if the room is being visited for the first time
        if (curr_i, curr_j) not in self.visited_rooms:
            room_embed[ROOM_ORDER.index((curr_i, curr_j))] = 1
            self.visited_rooms.append((curr_i,curr_j))

        if self.room_reward_task_vector:
            pickup = np.concatenate([pickup, room_embed],dtype=np.uint8)

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
        obs['train_tasks'] = self.train_tasks #we add train_tasks to the observation!!!

        if self.verbosity==1:
            print("Pickup: " + str(pickup))

        return obs, reward, done, info

#example of the env
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
        room_reward=.25,
        room_reward_task_vector=True,
        two_rooms=True
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