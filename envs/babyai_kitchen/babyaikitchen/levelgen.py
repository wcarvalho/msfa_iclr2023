import numpy as np

from gym import spaces


from gym_minigrid.minigrid import Grid, WorldObj
from babyai.levels.levelgen import RoomGridLevel, RejectSampling


from envs.babyai_kitchen.world import Kitchen
import envs.babyai_kitchen.tasks


TILE_PIXELS = 32


class KitchenLevel(RoomGridLevel):
    """
    """
    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        # locked_room_prob=0,
        unblocking=False,
        random_object_state=False,
        objects = [],
        actions = ['left', 'right', 'forward', 'pickup_container', 'pickup_contents', 'place', 'toggle', 'slice'],
        load_actions_from_tasks=False,
        task_kinds=['slice', 'clean', 'cook'],
        valid_tasks=[],
        taskarg_options=None,
        instr_kinds=['action'], # IGNORE. not implemented
        use_subtasks=False, # IGNORE. not implemented
        use_time_limit=True,
        tile_size=8,
        rootdir='.',
        distant_vision=False,
        agent_view_size=7,
        seed=None,
        verbosity=0,
        **kwargs,
    ):
        self.num_dists = num_dists
        # self.locked_room_prob = locked_room_prob
        self.use_time_limit = use_time_limit
        self.unblocking = unblocking
        self.valid_tasks = list(valid_tasks)
        if isinstance(task_kinds, list):
            self.task_kinds = task_kinds
        elif isinstance(task_kinds, str):
            self.task_kinds = [task_kinds]
        else:
            RuntimeError(f"Don't know how to read task kind(s): {str(task_kinds)}")
        self.instr_kinds = instr_kinds
        self.random_object_state = random_object_state
        self.use_subtasks = use_subtasks
        self.taskarg_options = taskarg_options

        self.verbosity = verbosity
        self.locked_room = None

        assert room_size >= 5, "otherwise can never place objects"
        agent_view_size = min(agent_view_size, room_size)
        if agent_view_size % 2 !=1:
            agent_view_size -= 1
        # ======================================================
        # agent view
        # ======================================================
        self.agent_view_width = agent_view_size
        if distant_vision:
            raise NotImplementedError
            # for far agent can see is length of room (-1 for walls)
            self.agent_view_height = room_size - 1
        else:
            self.agent_view_height = agent_view_size

        # ======================================================
        # setup env
        # ======================================================
        # define the dynamics of the objects with kitchen

        self.kitchen = Kitchen(objects=objects, tile_size=tile_size, rootdir=rootdir, verbosity=verbosity)
        self.check_task_actions = False

        # to avoid checking task during reset of initialization
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed,
            agent_view_size=self.agent_view_height,
            **kwargs,
        )
        self.check_task_actions = True

        # ======================================================
        # action space
        # ======================================================
        if load_actions_from_tasks:
            raise RuntimeError("Always use those set by __init__.")
            actions = self.load_actions_from_tasks(task_kinds)
        self.actions = {action:idx for idx, action in enumerate(actions, start=0)}
        self.idx2action = {idx:action for idx, action in enumerate(actions, start=0)}
        self.action_names = actions
        self.action_space = spaces.Discrete(len(self.actions))

        # ======================================================
        # observation space
        # ======================================================
        # potentially want to keep square and just put black for non-visible?
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_height, self.agent_view_width, 3),
            dtype='uint8'
        )

    def load_actions_from_tasks(self, task_kinds):
        actions = set()
        for kind in task_kinds:
            kind = kind.lower()

            if kind == "none": 
                continue

            task_class = envs.babyai_kitchen.tasks.TASKS[kind]
            task_actions = task_class.task_actions()

            for task_action in task_actions:
                if task_action in ["pickup", 'pickup_and']:
                    actions.add('pickup_contents')
                    actions.add('pickup_container')
                else:
                    actions.add(task_action)

        return ['left', 'right', 'forward'] + list(actions)

    # ======================================================
    # functions for generating grid + objeccts
    # ======================================================
    def _gen_grid(self, *args, **kwargs):
        """dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid are pretty confusing so just call base _gen_grid function to generate grid.
        """
        super(RoomGridLevel, self)._gen_grid(*args, **kwargs)

    def add_objects(self, task=None, num_distractors=10):
        """
        - if have task, place task objects
        
        Args:
            task (None, optional): Description
            num_distactors (int, optional): Description
        """
        placed_objects = set()
        # import ipdb; ipdb.set_trace()
        # first place task objects
        if task is not None:
            for obj in task.task_objects:
                self.place_in_room(0, 0, obj)
                placed_objects.add(obj.type)
                if self.verbosity > 1:
                    print(f"Added task object: {obj.type}")

        # if number of left over objects is less than num_distractors, set as that
        # possible_space = (self.grid.width - 2)*(self.grid.height - 2)
        num_leftover_objects = len(self.kitchen.objects)-len(placed_objects)
        num_distractors = min(num_leftover_objects, num_distractors)

        if len(placed_objects) == 0:
            num_distractors = max(num_distractors, 1)

        distractors_added = []
        num_tries = 0

        while len(distractors_added) < num_distractors:
            # infinite loop catch
            num_tries += 1
            if num_tries > 1000:
                raise RuntimeError("infinite loop in `add_objects`")

            # sample objects
            random_object = np.random.choice(self.kitchen.objects)

            # if already added, try again
            if random_object.type in placed_objects:
                continue

            self.place_in_room(0, 0, random_object)
            distractors_added.append(random_object.type)
            placed_objects.add(random_object.type)
            if self.verbosity > 1:
                print(f"Added distractor: {random_object.type}")

    # ======================================================
    # functions for generating and validating tasks
    # ======================================================
    def rand_task(
        self,
        task_kinds,
        instr_kinds,
        use_subtasks,
        depth=0
        ):

        if use_subtasks:
            raise RuntimeError("Don't know how to have subtask rewards")

        instruction_kind = np.random.choice(instr_kinds)

        if instruction_kind == 'action':
            task_kind = np.random.choice(task_kinds)
            task_kind = task_kind.lower()
            if task_kind == 'none':
                task = None
            else:
                task_class = envs.babyai_kitchen.tasks.TASKS[task_kind]
                task = task_class(
                    env=self.kitchen,
                    argument_options=self.taskarg_options)
        else:
            raise RuntimeError(f"Instruction kind not supported: '{instruction_kind}'")

        return task

    def generate_task(self):
        """copied from babyai.levels.levelgen:LevelGen.gen_mission
        """

        # connect all rooms
        self.connect_all()

        # reset kitchen objects
        self.kitchen.reset(randomize_states=self.random_object_state)

        # Generate random instructions
        task = self.rand_task(
            task_kinds=self.task_kinds,
            instr_kinds=self.instr_kinds,
            use_subtasks=self.use_subtasks,
        )
        if self.valid_tasks:
            idx = 0
            while not task.instruction in self.valid_tasks:
                task = self.rand_task(
                    task_kinds=self.task_kinds,
                    instr_kinds=self.instr_kinds,
                    use_subtasks=self.use_subtasks,
                )
                idx += 1
                if idx > 1000:
                    raise RuntimeError("infinite loop sampling possible task")


        self.add_objects(task=task, num_distractors=self.num_dists)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # self.unblocking==True means agent may need to unblock. don't check
        # self.unblocking==False means agent does not need to unblock. check
        if not self.unblocking:
            self.check_objs_reachable()


        return task

    def validate_task(self, task):
        if task is not None and self.check_task_actions:
            task.check_actions(self.action_names)

    def reset_task(self):
        """copied from babyai.levels.levelgen:RoomGridLevel._gen_drid
        - until success:
            - generate grid
            - generate task
                - generate objects
                - place object
                - generate language instruction
            - validate instruction
        """
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        tries = 0
        while True:
            tries += 1
            if tries > 1000:
                raise RuntimeError("can't sample task???")
            try:
                # generate grid of observation
                self._gen_grid(width=self.width, height=self.height)

                # Generate the task
                task = self.generate_task()

                # Validate the task
                self.validate_task(task)


            except RecursionError as error:
                print(f'Timeout during mission generation:{tries}/1000\n', error)
                continue

            except RejectSampling as error:
                #print('Sampling rejected:', error)
                continue

            break

        return task

    # ======================================================
    # reset, step used by gym
    # ======================================================
    def reset(self, **kwargs):
        """Copied from: 
        - gym_minigrid.minigrid:MiniGridEnv.reset
        - babyai.levels.levelgen:RoomGridLevel.reset
        the dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid were pretty confusing so I rewrote the base reset function.
        """
        # ======================================================
        # copied from: gym_minigrid.minigrid:MiniGridEnv.reset
        # ======================================================
        # reset current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # -----------------------
        # generate:
        # - grid
        # - objects
        # - agent location
        # - instruction
        # -----------------------
        # when call reset during initialization, don't load
        self.task = self.reset_task()
        if self.task is not None:
            self.surface = self.task.surface(self)
            self.mission = self.surface

            reward, done = self.task.check_status()
            if done:
                raise RuntimeError(f"`{self.mission}` started off as done")

            # make sure all task objects are on grid
            for obj in self.task.task_objects:
                assert obj.init_pos is not None
                assert obj.cur_pos is not None
                assert np.all(obj.init_pos == obj.cur_pos)
                assert self.grid.get(*obj.init_pos) is not None

        else:
            self.surface = self.mission = "No task"

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        # updating carrying in kitchen env just in case
        self.kitchen.update_carrying(self.carrying)
        # ======================================================
        # copied from babyai.levels.levelgen:RoomGridLevel.reset
        # ======================================================
        # # Recreate the verifier
        # if self.task:
        #     import ipdb; ipdb.set_trace()
        #     self.task.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = int(self.room_size ** 2)
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        if self.task:
            num_navs = self.task.num_navs
        else:
            num_navs = 1
        self.max_steps = num_navs * nav_time_maze

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

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        object_infront = self.grid.get(*fwd_pos)


        # Rotate left
        action_info = None
        interaction = False
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
            # if object_infront != None and object_infront.type == 'goal':
            #     done = True
            #     reward = self._reward()
            # if object_infront != None and object_infront.type == 'lava':
            #     done = True
        else:
            action_info = self.kitchen.interact(
                action=self.idx2action[int(action)],
                object_infront=object_infront,
                fwd_pos=fwd_pos,
                grid=self.grid,
                env=self, # only used for backwards compatibility with toggle
            )
            self.carrying = self.kitchen.carrying
            interaction = True

        step_info = self.kitchen.step()

        if self.verbosity > 1:
            from pprint import pprint
            print('='*50)
            obj_type = object_infront.type if object_infront else None
            print(self.idx2action[int(action)], obj_type)
            pprint(action_info)
            print('-'*10, 'Env Info', '-'*10)
            print("Carrying:", self.carrying)
            if self.task is not None:
                print(f"task objects:")
                pprint(self.task.task_objects)
            else:
                print(f"env objects:")
                pprint(self.kitchen.objects)

        # ======================================================
        # copied from RoomGridLevel
        # ======================================================

        # If we've successfully completed the mission
        info = {'success': False}
        if self.task is not None:
            reward, done = self.task.check_status()

            if done:
                info['success'] = True

        # if past step count, done
        if self.step_count >= self.max_steps and self.use_time_limit:
            done = True

        obs = self.gen_obs()
        return obs, reward, done, info



    # ======================================================
    # rendering functions
    # ======================================================

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        width, height, channels = obs.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                obj_idx, color_idx, state = obs[i, j]
                if obj_idx < 11:
                    object = WorldObj.decode(obj_idx, color_idx, state)
                    # vis_mask[i, j] = (obj_idx != OBJECT_TO_IDX['unseen'])
                else:
                    object = self.kitchen.objectid2object.get(obj_idx, None)
                if object:
                    grid.set(i, j, object)





        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img
