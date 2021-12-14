from envs.babyai_kitchen.world import Kitchen

def load_kitchen_tasks(tasks, kitchen=None, kitchen_kwargs={}):
    if len(tasks) == 0 or tasks is None: return [], []
    supported_object_types = set([
            'food',
            'utensil',
            'container',
        ])
    supported_task_types = dict(
        chill='chill x',
        slice='slice x',
        clean='clean x',
        heat='heat x',
        place='place x in y',
        cook='cook with y',
        )
    kitchen = kitchen or Kitchen(**kitchen_kwargs)
    _tasks = []
    _task_kinds = []
    for task_dict in tasks:
        if isinstance(task_dict, str):
            print("Warning: can't check if str is available. for checking use dict")
            _tasks.append(task_dict)
        elif isinstance(task_dict, dict):
            task = task_dict['task']
            _task_kinds.append(task)
            object = task_dict.get('object', None)
            remove = task_dict.get('remove', [])
            remove = remove if isinstance(remove, list) else [remove]

            form = supported_task_types[task]
            if object is not None:
                if isinstance(object, list):
                    objects = kitchen.objects_by_type(object)
                else:
                    if object in supported_object_types:
                        objects = kitchen.objects_by_type(object, prop='object_type')
                    else:
                        objects = kitchen.objects_by_type(object)
            else:
                raise NotImplementedError

            objects = list(filter(lambda o: not o.name in remove, objects))

            for o in objects:
                t = form.replace('x', o.name)

                if task in ['place', 'cook']:
                    container = task_dict.get("container", None)
                    if container:
                        t = t.replace('y', container)
                    else:
                        raise NotImplementedError

                _tasks.append(t)

        else:
            raise NotImplementedError(task_dict)


    return _tasks, _task_kinds



