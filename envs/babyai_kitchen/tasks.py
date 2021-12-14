import numpy as np
from babyai.levels.verifier import Instr
from envs.babyai_kitchen.world import Kitchen

def get_matching_objects(env, object_types=None, matchfn=None):
    """Get objects matching conditions
    
    Args:
        env (Kitchen): environment to select objects in
        object_types (None, optional): list of object types to sample from
        matchfn (TYPE, optional): criteria on objects to use for selecting
            options
    
    Returns:
        list: objects
    """
    if object_types is None and matchfn is None:
        return []

    if object_types:
        return env.objects_by_type(object_types)
    else:
        return [o for o in env.objects if matchfn(o)]


class KitchenTask(Instr):
    """docstring for KitchenTasks"""
    def __init__(self, env, argument_options=None):
        super(KitchenTask, self).__init__()
        self.argument_options = argument_options or dict(x=[])
        self._task_objects = []
        self.env = env
        self.instruction = self.generate()

    def generate(self):
        raise NotImplemented

    @property
    def task_objects(self):
        return self._task_objects

    def surface(self, *args, **kwargs):
        return self.instruction

    @property
    def num_navs(self): return 1

    def __repr__(self):
        string = self.instruction
        if self.task_objects:
            for object in self.task_objects:
                string += "\n" + str(object)

        return string

    def check_status(self):
        return False, False

    def check_actions(self, actions):
        for action in self.task_actions():
            if action == 'pickup':
                assert 'pickup_contents' in actions or 'pickup_container' in actions
            elif action == 'pickup_and':
                assert 'pickup_contents' in actions and 'pickup_container' in actions
            else:
                assert action in actions

    @staticmethod
    def task_actions():
        return [
            'toggle',
            'pickup_and',
            'place'
            ]

# ======================================================
# Length = 1
# ======================================================
class PickupTask(KitchenTask):
    @property
    def abstract_rep(self):
        return 'pickup x'

    def generate(self):
        # which option to pickup
        pickup_objects = get_matching_objects(self.env,
            object_types=self.argument_options.get('x', []),
            matchfn=lambda o:o.pickupable)

        self.pickup_object = np.random.choice(pickup_objects)

        self._task_objects = [
            self.pickup_object, 
        ]

        return self.abstract_rep.replace('x', self.pickup_object.type)

    def check_status(self):
        if self.env.carrying:
            # let's any match fit, not just the example used for defining the task. 
            # e.g., if multiple pots, any pot will work inside container
            done = reward = self.env.carrying.type == self.pickup_object.type
        else:
            done = reward = False

        return reward, done


# ======================================================
# Length = 2
# ======================================================

class HeatTask(KitchenTask):
    @property
    def abstract_rep(self):
        return 'heat x'

    def generate(self):
        self.stove = self.env.objects_by_type(['stove'])[0]

        x_options = self.argument_options.get('x', [])
        if x_options:
            objects_to_heat = self.env.objects_by_type(x_options)
        else:
            objects_to_heat = self.env.objects_by_type(self.stove.can_contain)
        
        self.object_to_heat = np.random.choice(objects_to_heat)


        self.object_to_heat.set_prop("temp", "room")
        self.stove.set_prop("temp", 'room')
        self.stove.set_prop("on", False)


        self._task_objects = [
            self.object_to_heat,
            self.stove,
        ]
        return self.abstract_rep.replace('x', self.object_to_heat.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_heat.state['temp'] == 'hot'
        return reward, done

class CleanTask(KitchenTask):

    @property
    def abstract_rep(self):
        return 'clean x'

    def generate(self):
        x_options = self.argument_options.get('x', [])
        if x_options:
            objects_to_clean = self.env.objects_by_type(x_options)
        else:
            objects_to_clean = self.env.objects_with_property(['dirty'])

        self.object_to_clean = np.random.choice(objects_to_clean)
        self.object_to_clean.set_prop('dirty', True)


        self.sink = self.env.objects_by_type(["sink"])[0]
        self.sink.set_prop('on', False)

        self._task_objects = [self.object_to_clean, self.sink]

        return self.abstract_rep.replace('x', self.object_to_clean.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_clean.state['dirty'] == False

        return reward, done

class SliceTask(KitchenTask):
    """docstring for SliceTask"""

    @property
    def abstract_rep(self):
        return 'slice x'

    def generate(self):
        x_options = self.argument_options.get('x', [])
        if x_options:
            objects_to_slice = self.env.objects_by_type(x_options)
        else:
            objects_to_slice = self.env.objects_with_property(['sliced'])

        self.object_to_slice = np.random.choice(objects_to_slice)
        self.object_to_slice.set_prop('sliced', False)

        self.knife = self.env.objects_by_type(["knife"])[0]

        self._task_objects = [self.object_to_slice, self.knife]
        return self.abstract_rep.replace('x', self.object_to_slice.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_slice.state['sliced'] == True

        return reward, done

    @staticmethod
    def task_actions():
        return [
            'slice',
            'pickup_and',
            'place'
            ]

class ChillTask(KitchenTask):
    """docstring for CookTask"""

    @property
    def abstract_rep(self):
        return 'chill x'

    def generate(self):
        self.fridge = self.env.objects_by_type(['fridge'])[0]

        x_options = self.argument_options.get('x', [])
        if x_options:
            objects_to_chill = self.env.objects_by_type(x_options)
        else:
            objects_to_chill = self.env.objects_by_type(self.fridge.can_contain)

        self.object_to_chill = np.random.choice(objects_to_chill)


        self.object_to_chill.set_prop("temp", "room")
        self.fridge.set_prop("temp", 'room')
        self.fridge.set_prop("on", False)


        self._task_objects = [
            self.object_to_chill,
            self.fridge,
        ]
        return self.abstract_rep.replace('x', self.object_to_chill.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_chill.state['temp'] == 'cold'

        return reward, done

class PickupCleanedTask(CleanTask):
    """docstring for CleanTask"""

    @property
    def abstract_rep(self):
        return 'pickup cleaned x'

    def generate(self):
        super().generate()

        return self.abstract_rep.replace('x', self.object_to_clean.name)

    def check_status(self):
        if self.env.carrying:
            clean = self.object_to_clean.state['dirty'] == False
            picked_up = self.env.carrying.type == self.object_to_clean.type
            reward = done = clean and picked_up
        else:
            done = reward = False

        return reward, done

class PickupSlicedTask(SliceTask):
    """docstring for SliceTask"""

    @property
    def abstract_rep(self):
        return 'pickup sliced x'

    def generate(self):
        super().generate()

        return self.abstract_rep.replace('x', self.object_to_slice.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        if self.env.carrying:
            sliced = self.object_to_slice.state['sliced'] == True
            picked_up = self.env.carrying.type == self.object_to_slice.type
            reward = done = sliced and picked_up
        else:
            done = reward = False

        return reward, done

    @staticmethod
    def task_actions():
        return [
            'slice',
            'pickup_and',
            'place'
            ]

class PickupChilledTask(ChillTask):
    """docstring for CookTask"""

    @property
    def abstract_rep(self):
        return 'pickup chilled x'

    def generate(self):
        super().generate()
        return self.abstract_rep.replace('x', self.object_to_chill.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        if self.env.carrying:
            chilled = self.object_to_chill.state['temp'] == 'cold'
            picked_up = self.env.carrying.type == self.object_to_chill.type
            reward = done = chilled and picked_up
        else:
            done = reward = False

        return reward, done


class PlaceTask(KitchenTask):

    @property
    def abstract_rep(self):
        return 'place x on y'

    def generate(self):
        # -----------------------
        # get possible containers/pickupable objects
        # -----------------------
        x_options = self.argument_options.get('x', [])
        y_options = self.argument_options.get('y', [])

        pickup_type_objs = get_matching_objects(self.env,
            object_types=x_options,
            matchfn=lambda o:o.pickupable)
        container_type_objs = get_matching_objects(self.env,
            object_types=y_options,
            matchfn=lambda o:o.is_container)

        if y_options and x_options:
            # pick container
            self.container = np.random.choice(container_type_objs)

            # pick objects which can be recieved by container
            pickup_types = [o.type for o in pickup_type_objs]
            pickup_types = [o for o in self.container.can_contain
                                if o in pickup_types]
            pickup_type_objs = self.env.objects_by_type(pickup_types)
            assert len(pickup_type_objs) > 0, "no match found"

            # pick 1 at random
            self.to_place = np.random.choice(pickup_type_objs)

        elif y_options:
            self.container = np.random.choice(container_type_objs)
            pickup_type_objs = self.env.objects_by_type(self.container.can_contain)
            self.to_place = np.random.choice(pickup_type_objs)

        elif x_options:
            # sample pickup first
            self.to_place = np.random.choice(pickup_type_objs)

            # restrict to wich can accept to_place
            container_type_objs = [o for o in self.env.objects 
                                    if o.accepts(self.to_place)]
            assert len(container_type_objs) > 0, "no match found"

            # pick 1 at random
            self.container = np.random.choice(container_type_objs)
        else:
            # pick container
            containers = [o for o in self.env.objects if o.is_container]
            self.container = np.random.choice(containers)

            # pick thing that can be placed inside
            pickup_type_objs = self.env.objects_by_type(self.container.can_contain)
            self.to_place = np.random.choice(pickup_type_objs)


        self._task_objects = [
            self.container, 
            self.to_place
        ]

        task = self.abstract_rep.replace('x', self.to_place.name)
        task = task.replace('y', self.container.name)
        return task

    def check_status(self):
        if self.container.contains:
            # let's any match fit, not just the example used for defining the task. 
            # e.g., if multiple pots, any pot will work inside container
            done = reward = self.container.contains.type == self.to_place.type
        else:
            done = reward = False

        return reward, done

    @staticmethod
    def task_actions():
        return [
            'pickup_and',
            'place'
            ]

# ======================================================
# length = 3
# ======================================================
class CookTask(KitchenTask):
    """docstring for CookTask"""

    @property
    def abstract_rep(self):
        return 'cook x with y'

    def generate(self):
        x_options = self.argument_options.get('x', [])
        y_options = self.argument_options.get('y', [])

        if x_options:
            objects_to_cook = self.env.objects_by_type(x_options)
        else:
            objects_to_cook = self.env.objects_with_property(['cooked']) # x

        if y_options:
            objects_to_cook_with = self.env.objects_by_type(y_options)
        else:
            objects_to_cook_with = self.env.objects_by_type(['pot', 'pan']) # y

        self.object_to_cook_on = self.env.objects_by_type(['stove'])[0]
        self.object_to_cook = np.random.choice(objects_to_cook)
        self.object_to_cook_with = np.random.choice(objects_to_cook_with)

        self.object_to_cook.set_prop("cooked", False)
        self.object_to_cook.set_prop("temp", 'room')
        self.object_to_cook_with.set_prop("dirty", False)
        self.object_to_cook_on.set_prop("on", False)


        self._task_objects = [
            self.object_to_cook,
            self.object_to_cook_with,
            self.object_to_cook_on
        ]

        task = self.abstract_rep.replace('x', self.object_to_cook.name)
        task = task.replace('y', self.object_to_cook_with.name)
        return task

    @property
    def num_navs(self): return 3

    def check_status(self):
        done = reward = self.object_to_cook.state['cooked'] == True

        return reward, done


TASKS=dict(
    pickup=PickupTask,
    place=PlaceTask,
    heat=HeatTask,
    clean=CleanTask,
    slice=SliceTask,
    chill=ChillTask,
    pickup_cleaned=PickupCleanedTask,
    pickup_sliced=PickupSlicedTask,
    pickup_chilled=PickupChilledTask,
    cook=CookTask,
)
