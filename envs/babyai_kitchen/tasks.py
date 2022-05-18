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

def remove_excluded(objects, exclude):
  return [o for o in objects if not o.type in exclude]

class KitchenTask(Instr):
  """docstring for KitchenTasks"""
  def __init__(self, env, argument_options=None, task_reps=None):
    super(KitchenTask, self).__init__()
    self.argument_options = argument_options or dict(x=[])
    self._task_objects = []
    self.env = env
    self.finished = False
    self._task_reps = task_reps
    self._task_name = 'kitchentask'
    self.instruction = self.generate()

  def generate(self, exclude=[]):
    raise NotImplemented

  def reset(self, exclude=[]):
    self.instruction = self.generate(exclude)

  @property
  def default_task_rep(self):
    raise NotImplemented

  @property
  def task_rep(self):
    if self._task_reps is not None:
      return self._task_reps.get(self.task_name, self.default_task_rep)
    else:
      return self.default_task_rep

  @property
  def task_name(self):
    return self._task_name

  @property
  def task_objects(self):
    return self._task_objects

  @property
  def task_types(self):
    return [o.type for o in self._task_objects]

  def surface(self, *args, **kwargs):
    return self.instruction

  def terminate(self, *args, **kwargs):
    self.finished = True

  def get_reward_done(self):
    if self.finished:
      return False, False
    else:
      return self.check_status()

  @property
  def num_navs(self): return 1

  def __repr__(self):
    string = self.instruction
    if self.task_objects:
        for object in self.task_objects:
            string += "\n" + str(object)

    return string

  def check_status(self): return False, False

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
  def default_task_rep(self):
    return 'pickup x'

  @property
  def task_name(self):
    return 'pickup'

  def generate(self, exclude=[]):
    # which option to pickup
    pickup_objects = get_matching_objects(self.env,
        object_types=self.argument_options.get('x', []),
        matchfn=lambda o:o.pickupable)
    pickup_objects = remove_excluded(pickup_objects, exclude)
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
  def default_task_rep(self): return 'turnon x'

  @property
  def task_name(self): return 'toggle'

  def generate(self, exclude=[]):

    x_options = self.argument_options.get('x', [])
    if x_options:
        totoggle_options = self.env.objects_by_type(x_options)
    else:
        totoggle_options = self.env.objects_with_property(['on'])

    totoggle_options = remove_excluded(totoggle_options, exclude)
    self.toggle = np.random.choice(totoggle_options)

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
    reward = done = self.toggle.state['on'] == True

    return reward, done

  def subgoals(self):
    return [
      ActionsSubgoal(
        goto=self.toggle, actions=['toggle']),
    ]

# ======================================================
# Length = 2
# ======================================================

class HeatTask(KitchenTask):
  @property
  def default_task_rep(self):
      return 'heat x'

  @property
  def task_name(self): return 'heat'

  def generate(self, exclude=[]):
    self.stove = self.env.objects_by_type(['stove'])[0]

    x_options = self.argument_options.get('x', [])
    if x_options:
        objects_to_heat = self.env.objects_by_type(x_options)
    else:
        objects_to_heat = self.env.objects_by_type(self.stove.can_contain)
    objects_to_heat = remove_excluded(objects_to_heat, exclude)
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
  def task_name(self): return 'clean'

  @property
  def default_task_rep(self):
    return 'clean x'

  def generate(self, exclude=[]):
    x_options = self.argument_options.get('x', [])
    exclude = ['sink']+exclude
    if x_options:
        objects_to_clean = self.env.objects_by_type(x_options)
    else:
        objects_to_clean = self.env.objects_with_property(['dirty'])

    objects_to_clean = remove_excluded(objects_to_clean, exclude)
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
  def task_name(self): return 'slice'

  @property
  def default_task_rep(self):
    return 'slice x'

  def get_options(self, exclude=[]):
    x_options = self.argument_options.get('x', [])
    if x_options:
        objects_to_slice = self.env.objects_by_type(x_options)
    else:
        objects_to_slice = self.env.objects_with_property(['sliced'])
    objects_to_slice = remove_excluded(objects_to_slice, exclude)
    return {'x': objects_to_slice}

  def generate(self, exclude=[]):

    objects_to_slice = self.get_options(exclude)['x']
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
  def task_name(self): return 'chill'

  @property
  def default_task_rep(self):
    return 'chill x'

  def generate(self, exclude=[]):
    self.fridge = self.env.objects_by_type(['fridge'])[0]

    x_options = self.argument_options.get('x', [])
    if x_options:
        objects_to_chill = self.env.objects_by_type(x_options)
    else:
        objects_to_chill = self.env.objects_by_type(self.fridge.can_contain)

    objects_to_chill = remove_excluded(objects_to_chill, exclude)
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
  def task_name(self): return 'pickup_cleaned'

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
  def task_name(self): return 'pickup_sliced'
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
  def task_name(self): return 'pickup_chilled'
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
  def task_name(self): return 'pickup_heated'
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
  def task_name(self): return 'place'
  @property
  def default_task_rep(self):
    return 'place x on y'

  def generate(self, exclude=[]):
    if exclude:
      raise NotImplementedError
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
  def task_name(self): return 'pickup_placed'
  @property
  def default_task_rep(self):
    return 'pickup x on y'

  def check_status(self):
    placed, placed = super().__check_status(self)
    carrying = pickedup(self.env, self.container)
    done = reward = carrying and placed
    return reward, done

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
  def task_name(self): return 'slice2'
  @property
  def default_task_rep(self):
    return 'slice x and slice y'

  def generate(self, exclude=[]):
    objects_to_slice = self.get_options(exclude)['x']

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
  def task_name(self): return 'toggle2'
  @property
  def default_task_rep(self):
    return 'turnon x and turnon y'

  def generate(self, exclude=[]):
    
    x_options = self.argument_options.get('x', [])
    if x_options:
        totoggle_options = self.env.objects_by_type(x_options)
    else:
        totoggle_options = self.env.objects_with_property(['on'])

    totoggle_options = remove_excluded(totoggle_options, exclude)
    assert len(totoggle_options) > 1

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

class CleanAndSliceTask(KitchenTask):
  """docstring for SliceTask"""
  def __init__(self, *args, **kwargs):

    self.clean_task = CleanTask(*args, **kwargs)
    self.slice_task = SliceTask(*args, **kwargs)
    super(CleanAndSliceTask, self).__init__(*args, **kwargs)

  @property
  def task_name(self): return 'clean_and_slice'
  @property
  def default_task_rep(self):
    part1 = self.clean_task.default_task_rep()
    part2 = self.slice_task.default_task_rep().replace("x", "y")
    return f"{part1} and {part2}"

  def generate(self, exclude=[]):
    slice_instr = self.slice_task.generate()
    clean_instr = self.clean_task.generate(exclude=['knife'])

    self._task_objects = self.clean_task.task_objects + \
      self.slice_task.task_objects
    instr =  f"{clean_instr} and {slice_instr}"

    return instr

  @property
  def num_navs(self): return 2

  def check_status(self):
    _, clean = self.clean_task.check_status()
    _, sliced = self.slice_task.check_status()
    
    reward = done = clean and sliced

    return reward, done

  def subgoals(self):
    subgoals = self.clean_task.subgoals()+self.slice_task.subgoals()
    return subgoals

class ToggleAndSliceTask(KitchenTask):
  """docstring for SliceTask"""
  def __init__(self, *args, **kwargs):
    self.toggle_task = ToggleTask(*args, **kwargs)
    self.slice_task = SliceTask(*args, **kwargs)
    super(ToggleAndSliceTask, self).__init__(*args, **kwargs)

  @property
  def task_name(self): return 'toggle_and_slice'
  @property
  def default_task_rep(self):
    part1 = self.toggle_task.default_task_rep()
    part2 = self.slice_task.default_task_rep().replace("x", "y")
    return f"{part1} and {part2}"

  def generate(self, exclude=[]):
    slice_instr = self.slice_task.generate()
    toggle_instr = self.toggle_task.generate()

    self._task_objects = self.toggle_task.task_objects + \
      self.slice_task.task_objects
    instr =  f"{toggle_instr} and {slice_instr}"

    return instr

  @property
  def num_navs(self): return 2

  def check_status(self):
    _, toggled = self.toggle_task.check_status()
    _, sliced = self.slice_task.check_status()
    
    reward = done = toggled and sliced

    return reward, done

  def subgoals(self):
    subgoals = self.toggle_task.subgoals()+self.slice_task.subgoals()
    return subgoals

class CleanAndToggleTask(KitchenTask):
  """docstring for CleanTask"""
  def __init__(self, *args, **kwargs):
    self.toggle_task = ToggleTask(*args, **kwargs)
    self.clean_task = CleanTask(*args, **kwargs)
    super(CleanAndToggleTask, self).__init__(*args, **kwargs)

  @property
  def task_name(self): return 'clean_and_toggle'
  @property
  def default_task_rep(self):
    part1 = self.toggle_task.default_task_rep()
    part2 = self.clean_task.default_task_rep().replace("x", "y")
    return f"{part1} and {part2}"

  def generate(self, exclude=[]):
    clean_instr = self.clean_task.generate()
    Toggle_instr = self.toggle_task.generate(exclude=self.clean_task.task_types)

    self._task_objects = self.toggle_task.task_objects + \
      self.clean_task.task_objects
    instr =  f"{Toggle_instr} and {clean_instr}"

    return instr

  @property
  def num_navs(self): return 2

  def check_status(self):
    _, toggled = self.toggle_task.check_status()
    _, cleand = self.clean_task.check_status()
    
    reward = done = toggled and cleand

    return reward, done

  def subgoals(self):
    subgoals = self.toggle_task.subgoals()+self.clean_task.subgoals()
    return subgoals

# ======================================================
# length = 3
# ======================================================
class CookTask(KitchenTask):
  """docstring for CookTask"""

  @property
  def task_name(self): return 'cook'
  @property
  def default_task_rep(self):
    return 'cook x with y'

  def generate(self, exclude=[]):
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
  def task_name(self): return 'pickup_cooked'
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
  def task_name(self): return 'place_sliced'
  @property
  def default_task_rep(self):
    return 'place sliced x on y'

  def generate(self, exclude=[]):
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

class CleanAndSliceAndToggleTask(KitchenTask):
  """docstring for SliceTask"""
  def __init__(self, *args, **kwargs):
    self.clean_task = CleanTask(*args, **kwargs)
    self.slice_task = SliceTask(*args, **kwargs)
    self.toggle_task = ToggleTask(*args, **kwargs)
    super(CleanAndSliceAndToggleTask, self).__init__(*args, **kwargs)

  @property
  def task_name(self): return 'clean_and_slice_and_toggle'
  @property
  def default_task_rep(self):
    part1 = self.clean_task.default_task_rep()
    part2 = self.slice_task.default_task_rep().replace("x", "y")
    part3 = self.toggle_task.default_task_rep().replace("x", "z")
    return f"{part1} and {part2} and {part3}"

  def generate(self, exclude=[]):
    slice_instr = self.slice_task.generate()
    clean_instr = self.clean_task.generate(exclude=['knife'])
    toggle_instr = self.toggle_task.generate(
      exclude=self.clean_task.task_types)

    self._task_objects = self.clean_task.task_objects + \
      self.slice_task.task_objects + self.toggle_task.task_objects
    instr =  f"{clean_instr} and {slice_instr} and {toggle_instr}"

    return instr

  @property
  def num_navs(self): return 3

  def check_status(self):
    _, clean = self.clean_task.check_status()
    _, sliced = self.slice_task.check_status()
    _, toggled = self.toggle_task.check_status()
    
    reward = done = clean and sliced and toggled

    return reward, done

  def subgoals(self):
    subgoals = self.clean_task.subgoals()+self.slice_task.subgoals()+self.toggle_task.subgoals()
    return subgoals


def all_tasks():
  return dict(
    pickup=PickupTask,
    toggle=ToggleTask,
    place=PlaceTask,
    heat=HeatTask,
    clean=CleanTask,
    slice=SliceTask,
    chill=ChillTask,
    slice2=Slice2Task,
    clean_and_slice=CleanAndSliceTask,
    clean_and_toggle=CleanAndToggleTask,
    toggle_and_slice=ToggleAndSliceTask,
    clean_and_slice_and_toggle=CleanAndSliceAndToggleTask,
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
