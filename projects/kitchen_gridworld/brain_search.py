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
  elif search == 'long_r2d1':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['long']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
      {
        "seed": tune.grid_search([3, 4]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['long']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      }
    ]
  elif search == 'long_baselines':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['long']),
        "embed_task_dim": tune.grid_search([4]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['usfa_lstm']),
        "setting": tune.grid_search(['long']),
        "embed_task_dim": tune.grid_search([4]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      }
    ]

  elif search == 'long_fixed':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([seed]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['long']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      } for seed in [1, 2, 3, 4]
    ]

  elif search == 'long_msf':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([seed]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['long']),
        "nmodules": tune.grid_search([2]),
        "module_task_dim": tune.grid_search([2]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      } for seed in [1, 2]] + [{
        "seed": tune.grid_search([seed]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['long']),
        "nmodules": tune.grid_search([2]),
        "module_task_dim": tune.grid_search([1]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      } for seed in [1, 2]
    ]


  elif search == 'long_msf2':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([3]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "struct_policy_input": tune.grid_search([True]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
      {
        "seed": tune.grid_search([4]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "struct_policy_input": tune.grid_search([True]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "nmodules": tune.grid_search([2]),
        "struct_policy_input": tune.grid_search([True]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
      {
        "seed": tune.grid_search([3, 2]),
        "group": tune.grid_search(['long_baselines']),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['long']),
        "samples_per_insert": tune.grid_search([6.0]),
        "reward_coeff": tune.grid_search([10.0]),
        "nmodules": tune.grid_search([2]),
        "struct_policy_input": tune.grid_search([True]),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      }
    ]

  else:
    raise NotImplementedError(search)

  return space, actor_label, evaluator_label