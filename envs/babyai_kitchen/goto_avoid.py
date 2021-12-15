"""

"""


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
        nobjects=12,
        **kwargs):

        self.object2reward = object2reward
        objects = object2reward.keys()
        self.nobjects = nobjects
        kitchen = Kitchen(
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


    def reset_task(self):
        # generate grid
        self._gen_grid(width=self.width, height=self.height)

        # connect all rooms
        self.connect_all()

        for idx in range(self.nobjects):
            choice = np.random.randint(len(self.default_objects))
            object = copy.deepcopy(self.default_objects[choice])
            self.place_in_room(0, 0, object)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break



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

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        object_infront = self.grid.get(*fwd_pos)


        # Rotate left
        action_info = None
        if action == self.actions.get('left', -1):
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.get('right', -1):
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.get('forward', -1):
            if object_infront == None or object_infront.can_overlap():
                self.agent_pos = fwd_pos
        else:
            if object_infront:
                # get reward
                if object_infront.type in self.object2reward:
                    reward = self.object2reward[object_infront.type]
                    self.grid.set(*fwd_pos, None)
                    # move object
                    self.place_in_room(0, 0, object_infront)

            else:
                pass

        # ======================================================
        # copied from RoomGridLevel
        # ======================================================
        info = {}
        # if past step count, done
        if self.step_count >= self.max_steps and self.use_time_limit:
            done = True

        obs = self.gen_obs()

        obs['mission'] = np.array(self.object2reward.values())

        return obs, reward, done, info

if __name__ == '__main__':
    import gym_minigrid.window
    import time
    from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
    import matplotlib.pyplot as plt 
    import cv2

    env = GotoAvoidEnv(
        room_size=10,
        agent_view_size=5,
        object2reward={
            "pan" : 1,
            "plates" : 0,
            "fork" : 0,
            "knife" : 0,
            },
        tile_size=12,
        nobjects=12,
        )
    env = RGBImgPartialObsWrapper(env, tile_size=12)

    def combine(full, partial):
        full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
        return np.concatenate((full_small, partial), axis=1)

    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)


    obs = env.reset()
    full = env.render('rgb_array', tile_size=12, highlight=True)
    window.set_caption(obs['mission'])
    window.show_img(combine(full, obs['image']))

    for step in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())
        full = env.render('rgb_array', tile_size=12, highlight=True)
        window.show_img(combine(full, obs['image']))
