from ray import tune

def get(search, agent=''):
  agent = agent or 'r2d1'
  actor_label=None
  evaluator_label=None

  if search == 'slice5':
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['place_sliced']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
    ]

  elif search == 'test12_baselines':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search([agent]),
      #   "setting": tune.grid_search([setting]),
      #   "struct_and": tune.grid_search([True]),
      #   "module_task_dim": tune.grid_search([1]),
      #   "max_number_of_steps": tune.grid_search([20_000_000]),
      # } for setting in ['clean_gen', 'cook_gen', 'toggle_gen', 'slice_gen']
      # {
      #   "seed": tune.grid_search([2, 3]),
      #   "agent": tune.grid_search(['usfa_lstm', 'r2d1']),
      #   "setting": tune.grid_search(['gen_toggle_pickup']),
      #   "struct_and": tune.grid_search([True]),
      #   "label": tune.grid_search(['v2']),
      #   "samples_per_insert": tune.grid_search([6.0]),
      #   "max_number_of_steps": tune.grid_search([5_000_000]),
      # },
      # {
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search([agent]),
      #   "setting": tune.grid_search(['gen_toggle_pickup']),
      #   "struct_and": tune.grid_search([True]),
      #   "samples_per_insert": tune.grid_search([6.0]),
      #   "reward_coeff": tune.grid_search([50.0, 10.0]),
      #   "nmodules": tune.grid_search([8, 16]),
      #   "max_number_of_steps": tune.grid_search([5_000_000]),
      # }
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['gen_toggle_pickup']),
        "struct_and": tune.grid_search([True]),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([50.0]),
        "nmodules": tune.grid_search([4, 8]),
        "struct_policy_input": tune.grid_search([True]),
        "eval_task_support": tune.grid_search(['train']),
        "max_number_of_steps": tune.grid_search([5_000_000]),
      }
    ]

  else:
    raise NotImplementedError(search)

  return space, actor_label, evaluator_label