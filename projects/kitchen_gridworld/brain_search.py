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
  elif search == 'long_baselines':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "max_number_of_steps": tune.grid_search([50_000_000]),
      } for agent in ['usfa_lstm', 'r2d1', 'msf', 'modr2d1']
    ]

  elif search == 'long_msf':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "struct_policy_input": tune.grid_search([True]),
        "max_number_of_steps": tune.grid_search([50_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "struct_policy_input": tune.grid_search([False]),
        "max_number_of_steps": tune.grid_search([50_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "nmodules": tune.grid_search([8]),
        "struct_policy_input": tune.grid_search([True]),
        "max_number_of_steps": tune.grid_search([50_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "nmodules": tune.grid_search([8]),
        "struct_policy_input": tune.grid_search([False]),
        "max_number_of_steps": tune.grid_search([50_000_000]),
      }
    ]

  elif search == 'test13_baselines':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['gen_toggle_pickup']),
        "struct_and": tune.grid_search([True]),
        "samples_per_insert": tune.grid_search([6.0]),
        "max_number_of_steps": tune.grid_search([5_000_000]),
      } for agent in ['usfa_lstm', 'r2d1', 'msf', 'r2d1_no_task']
      # {
      #   "seed": tune.grid_search([1, 2, 3]),
      #   "agent": tune.grid_search([agent]),
      #   "setting": tune.grid_search(['gen_toggle_pickup_slice']),
      #   "struct_and": tune.grid_search([True]),
      #   "samples_per_insert": tune.grid_search([6.0]),
      #   "max_number_of_steps": tune.grid_search([5_000_000]),
      # } for agent in ['usfa_lstm', 'r2d1', 'msf', 'r2d1_no_task']
    ]

  elif search == 'test13_msf':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['gen_toggle_pickup']),
        "struct_and": tune.grid_search([True]),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([50.0]),
        "nmodules": tune.grid_search([4]),
        "struct_policy_input": tune.grid_search([True]),
        "eval_task_support": tune.grid_search(['train']),
        "max_number_of_steps": tune.grid_search([5_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['gen_toggle_pickup']),
        "struct_and": tune.grid_search([True]),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "nmodules": tune.grid_search([4]),
        "struct_policy_input": tune.grid_search([True]),
        "eval_task_support": tune.grid_search(['train']),
        "max_number_of_steps": tune.grid_search([5_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['gen_toggle_pickup']),
        "struct_and": tune.grid_search([True]),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([50.0]),
        "nmodules": tune.grid_search([4]),
        "struct_policy_input": tune.grid_search([False]),
        "eval_task_support": tune.grid_search(['train']),
        "max_number_of_steps": tune.grid_search([5_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['gen_toggle_pickup']),
        "struct_and": tune.grid_search([True]),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "nmodules": tune.grid_search([4]),
        "struct_policy_input": tune.grid_search([False]),
        "eval_task_support": tune.grid_search(['train']),
        "max_number_of_steps": tune.grid_search([5_000_000]),
      }
    ]

  else:
    raise NotImplementedError(search)

  return space, actor_label, evaluator_label