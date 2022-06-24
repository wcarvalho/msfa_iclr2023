from ray import tune

def get(search, agent=''):
  agent = agent or 'r2d1'
  actor_label='actor_struct_and'
  evaluator_label='eval_struct_and'

  if search == 'slice5':
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['place_sliced']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
    ]


  elif search == 'test11_baselines':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['toggle_gen']),
        "struct_and": tune.grid_search([True]),
        "task_reset_behavior": tune.grid_search([task_reset_behavior]),
        "max_number_of_steps": tune.grid_search([2_000_000]),
      } for task_reset_behavior in ['none', 'remove', 'respawn']
    ]

  elif search == 'test11_toggle':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['toggle_gen']),
      #   "struct_and": tune.grid_search([True]),
      #   "task_reps": tune.grid_search(['object_verbose']),
      #   "label": tune.grid_search(['v4']),
      #   "bag_of_words": tune.grid_search([True]),
      #   "embed_task_dim": tune.grid_search([0, 16]),
      #   "max_number_of_steps": tune.grid_search([1_000_000]),
      # },
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['toggle_gen']),
        "struct_and": tune.grid_search([True]),
        "task_reps": tune.grid_search(['object_verbose']),
        "label": tune.grid_search(['v5']),
        "bag_of_words": tune.grid_search([False]),
        "word_compress": tune.grid_search(['last', 'sum']),
        "embed_task_dim": tune.grid_search([0, 16]),
        "module_task_dim": tune.grid_search([0]),
        "max_number_of_steps": tune.grid_search([1_000_000]),
      },
    ]

  else:
    raise NotImplementedError(search)

  return space, actor_label, evaluator_label