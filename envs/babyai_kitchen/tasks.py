import numpy as np
from babyai.levels.verifier import Instr
from envs.babyai_kitchen.world import Kitchen
from envs.babyai_kitchen.types import ActionsSubgoal

from babyai.bot import Bot, GoNextToSubgoal
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                            GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)


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

def pickedup(env, obj):
  return env.carrying.type == obj.type



# ======================================================
# Length = 1
# ======================================================
class PickupTask(KitchenTask):

    @property
    def default_task_rep(self):
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

        return self.task_rep.replace('x', self.pickup_object.type)

    def check_status(self):
        if self.env.carrying:
            # let's any match fit, not just the example used for defining the task. 
            # e.g., if multiple pots, any pot will work inside container
            done = reward = self.env.carrying.type == self.pickup_object.type
        else:
            done = reward = False

        return reward, done


    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.pickup_object, actions=['pickup_contents'])
      ]

class ToggleTask(KitchenTask):
    @property
    def default_task_rep(self):
        return 'Turnon x'

    def generate(self):

        x_options = self.argument_options.get('x', [])
        if x_options:
            totoggle_options = self.env.objects_by_type(x_options)
        else:
            totoggle_options = self.env.objects_with_property(['on'])

        self.toggle = np.random.choice(totoggle_options, 1, replace=False)

        self.toggle.set_prop("on", False)

        self._task_objects = [
            self.toggle,
        ]
        instr = self.task_rep.replace(
          'x', self.toggle.name)

        return instr

    @property
    def num_navs(self): return 2

    def check_status(self):
        toggle1 = self.toggle.state['on'] == True
        toggle2 = self.toggle2.state['on'] == True
        reward = done = toggle1 and toggle2
        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.toggle, actions=['toggle']),
        ActionsSubgoal(
          goto=self.toggle2, actions=['toggle']),
      ]

# ======================================================
# Length = 2
# ======================================================

class HeatTask(KitchenTask):
    @property
    def default_task_rep(self):
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
        return self.task_rep.replace('x', self.object_to_heat.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_heat.state['temp'] == 'hot'
        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_heat, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.stove, actions=['place', 'toggle'])
      ]

class CleanTask(KitchenTask):

    @property
    def default_task_rep(self):
        return 'clean x'

    def generate(self):
        x_options = self.argument_options.get('x', [])
        if x_options:
            objects_to_clean = self.env.objects_by_type(x_options)
        else:
            objects_to_clean = self.env.objects_with_property(['dirty'])
            objects_to_clean = [o for o in objects_to_clean if o.type != 'sink']

        self.object_to_clean = np.random.choice(objects_to_clean)
        self.object_to_clean.set_prop('dirty', True)


        self.sink = self.env.objects_by_type(["sink"])[0]
        self.sink.set_prop('on', False)

        self._task_objects = [self.object_to_clean, self.sink]

        return self.task_rep.replace('x', self.object_to_clean.name)

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_clean, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.sink, actions=['place', 'toggle'])
      ]

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_clean.state['dirty'] == False

        return reward, done

class SliceTask(KitchenTask):
    """docstring for SliceTask"""

    @property
    def default_task_rep(self):
        return 'slice x'

    def get_options(self):
      x_options = self.argument_options.get('x', [])
      if x_options:
          objects_to_slice = self.env.objects_by_type(x_options)
      else:
          objects_to_slice = self.env.objects_with_property(['sliced'])
      return {'x': objects_to_slice}

    def generate(self):

        objects_to_slice = self.get_options()['x']
        self.object_to_slice = np.random.choice(objects_to_slice)
        self.object_to_slice.set_prop('sliced', False)

        self.knife = self.env.objects_by_type(["knife"])[0]

        self._task_objects = [self.object_to_slice, self.knife]
        return self.task_rep.replace('x', self.object_to_slice.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_slice.state['sliced'] == True

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.knife, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.object_to_slice, actions=['slice'])
      ]

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
    def default_task_rep(self):
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
        return self.task_rep.replace('x', self.object_to_chill.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_chill.state['temp'] == 'cold'

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_chill, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.fridge, actions=['place', 'toggle'])
      ]

class PickupCleanedTask(CleanTask):
    """docstring for CleanTask"""

    @property
    def default_task_rep(self):
        return 'pickup cleaned x'

    def check_status(self):
        if self.env.carrying:
            clean = self.object_to_clean.state['dirty'] == False
            picked_up = self.env.carrying.type == self.object_to_clean.type
            reward = done = clean and picked_up
        else:
            done = reward = False

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_clean, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.sink, actions=['place', 'toggle', 'pickup_contents'])
      ]

class PickupSlicedTask(SliceTask):
    """docstring for SliceTask"""

    @property
    def default_task_rep(self):
        return 'pickup sliced x'

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

    def subgoals(self):
      # TODO: rotate and try to place is a hack that should be replaced
      return [
        ActionsSubgoal(
          goto=self.knife, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.object_to_slice, actions=['slice', *(['left', 'place']*4), 'pickup_contents'])
      ]

class PickupChilledTask(ChillTask):
    """docstring for CookTask"""

    @property
    def default_task_rep(self):
        return 'pickup chilled x'

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

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_chill, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.fridge, actions=['place', 'toggle', 'pickup_contents'])
      ]

class PickupHeatedTask(HeatTask):
    """docstring for CookTask"""

    @property
    def default_task_rep(self):
        return 'pickup heated x'

    @property
    def num_navs(self): return 2

    def check_status(self):
        if self.env.carrying:
            heated = self.object_to_heat.state['temp'] == 'hot'
            picked_up = self.env.carrying.type == self.object_to_heat.type
            reward = done = heated and picked_up
        else:
            done = reward = False

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_heat, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.stove, actions=['place', 'toggle', 'pickup_contents'])
      ]

class PlaceTask(KitchenTask):

    @property
    def default_task_rep(self):
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

        task = self.task_rep.replace('x', self.to_place.name)
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

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.to_place, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.container, actions=['place'])
      ]

class PickupPlacedTask(KitchenTask):

    @property
    def default_task_rep(self):
        return 'pickup x on y'

    def check_status(self):
      placed, placed = super().__check_status(self)
      carrying = pickedup(self.env, self.container)
      done = reward = carrying and placed
      return reward, done

    @staticmethod
    def task_actions():
        return [
            'pickup_and',
            'place'
            ]

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.to_place, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.container, actions=['place', 'pickup_container'])
      ]

class Slice2Task(SliceTask):
    """docstring for SliceTask"""

    @property
    def default_task_rep(self):
        return 'slice x and y'

    def generate(self):
        objects_to_slice = self.get_options()['x']


        choices = np.random.choice(objects_to_slice, 2, replace=False)

        self.object_to_slice = choices[0]
        self.object_to_slice2 = choices[1]

        self.object_to_slice.set_prop('sliced', False)
        self.object_to_slice2.set_prop('sliced', False)

        self.knife = self.env.objects_by_type(["knife"])[0]

        self._task_objects = [self.object_to_slice, self.object_to_slice2, self.knife]

        instr =  self.task_rep.replace(
          'x', self.object_to_slice.name).replace(
          'y', self.object_to_slice2.name)

        return instr

    @property
    def num_navs(self): return 2

    def check_status(self):
        sliced1  = self.object_to_slice.state['sliced'] == True
        sliced2 = self.object_to_slice2.state['sliced'] == True

        reward = done = sliced1 and sliced2

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.knife, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.object_to_slice, actions=['slice']),
        ActionsSubgoal(
          goto=self.object_to_slice2, actions=['slice'])
      ]

    @staticmethod
    def task_actions():
        return [
            'slice',
            'pickup_and',
            'place'
            ]

class Toggle2Task(KitchenTask):
    @property
    def default_task_rep(self):
        return 'Turnon x and y'

    def generate(self):
        
        x_options = self.argument_options.get('x', [])
        if x_options:
            totoggle_options = self.env.objects_by_type(x_options)
        else:
            totoggle_options = self.env.objects_with_property(['on'])

        choices = np.random.choice(totoggle_options, 2, replace=False)

        self.toggle1 = choices[0]
        self.toggle2 = choices[1]

        self.toggle1.set_prop("on", False)
        self.toggle2.set_prop("on", False)

        self._task_objects = [
            self.toggle1,
            self.toggle2,
        ]
        instr = self.task_rep.replace(
          'x', self.toggle1.name).replace(
          'y', self.toggle2.name)

        return instr

    @property
    def num_navs(self): return 2

    def check_status(self):
        toggle1 = self.toggle1.state['on'] == True
        toggle2 = self.toggle2.state['on'] == True
        reward = done = toggle1 and toggle2
        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.toggle1, actions=['toggle']),
        ActionsSubgoal(
          goto=self.toggle2, actions=['toggle']),
      ]

# ======================================================
# length = 3
# ======================================================
class CookTask(KitchenTask):
    """docstring for CookTask"""

    @property
    def default_task_rep(self):
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

        task = self.task_rep.replace('x', self.object_to_cook.name)
        task = task.replace('y', self.object_to_cook_with.name)
        return task

    @property
    def num_navs(self): return 3

    def check_status(self):
        done = reward = self.object_to_cook.state['cooked'] == True

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.object_to_cook, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.object_to_cook_with, actions=['place', 'pickup_container']),
        ActionsSubgoal(
          goto=self.object_to_cook_on, actions=['place', 'toggle'])
      ]

class PickupCookedTask(CookTask):

  @property
  def default_task_rep(self):
      return 'pickup cooked x'


  @property
  def num_navs(self): return 3

  def check_status(self):
      _, done = super().check_status()
      if self.env.carrying:
          picked_up = self.env.carrying.type == self.object_to_cook.type
          reward = done = done and picked_up
      else:
          done = reward = False

      return reward, done


  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.object_to_cook, actions=['pickup_contents']),
      ActionsSubgoal(
        goto=self.object_to_cook_with, actions=['place', 'pickup_container']),
      ActionsSubgoal(
        goto=self.object_to_cook_on, actions=['place', 'toggle', 'pickup_contents'])
    ]

class PlaceSlicedTask(SliceTask):
    """docstring for SliceTask"""

    @property
    def default_task_rep(self):
        return 'place sliced x on y'

    def generate(self):
        # -----------------------
        # x= object to slice
        # -----------------------
        x_options = self.argument_options.get('x', [])
        if x_options:
            objects_to_slice = self.env.objects_by_type(x_options)
        else:
            objects_to_slice = self.env.objects_with_property(['sliced'])
        self.object_to_slice = np.random.choice(objects_to_slice)
        self.object_to_slice.set_prop('sliced', False)

        # -----------------------
        # knife
        # -----------------------
        self.knife = self.env.objects_by_type(["knife"])[0]


        # -----------------------
        # y = container
        # -----------------------

        # restrict to wich can accept to_place
        container_type_objs = [o for o in self.env.objects 
                                if o.accepts(self.object_to_slice)]
        assert len(container_type_objs) > 0, "no match found"

        # pick 1 at random
        self.container = np.random.choice(container_type_objs)


        self._task_objects = [self.object_to_slice, self.knife, self.container]
        return self.task_rep.replace(
          'x', self.object_to_slice.name).replace(
          'y', self.container.name)

    @property
    def num_navs(self): return 2

    def check_status(self):
        if self.container.contains:
            # let's any match fit, not just the example used for defining the task. 
            # e.g., if multiple pots, any pot will work inside container
            object_sliced = self.object_to_slice.state['sliced'] == True
            placed = self.container.contains.type == self.object_to_slice.type
            done = reward = object_sliced and placed
        else:
            done = reward = False

        return reward, done

    def subgoals(self):
      return [
        ActionsSubgoal(
          goto=self.knife, actions=['pickup_contents']),
        ActionsSubgoal(
          goto=self.object_to_slice, actions=['slice', *(['left', 'place']*4), 'pickup_contents']),
        ActionsSubgoal(
          goto=self.container, actions=['place']),
      ]

    @staticmethod
    def task_actions():
        return [
            'slice',
            'pickup_and',
            'place'
            ]

def all_tasks():
  return dict(
      pickup=PickupTask,
      place=PlaceTask,
      heat=HeatTask,
      clean=CleanTask,
      slice=SliceTask,
      chill=ChillTask,
      slice2=Slice2Task,
      toggle2=Toggle2Task,
      pickup_cleaned=PickupCleanedTask,
      pickup_sliced=PickupSlicedTask,
      pickup_chilled=PickupChilledTask,
      pickup_heated=PickupHeatedTask,
      cook=CookTask,
      pickup_cooked=PickupCookedTask,
      pickup_placed=PickupPlacedTask,
      place_sliced=PlaceSlicedTask,
  )
