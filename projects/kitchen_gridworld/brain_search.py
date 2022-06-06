from ray import tune

def get(search):
  if search == 'slice4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
        "setting": tune.grid_search(['slice']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]

  elif search == 'cook4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['usfa_lstm']),
        "setting": tune.grid_search(['cook']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]

  elif search == 'similar4':
    """
    Next:
    """
    space = [
      # { # 6
      #   "seed": tune.grid_search([2, 3]),
      #   "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
      #   "setting": tune.grid_search(['similar']),
      # },
      { # 6
        "seed": tune.grid_search([2, 3]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['similar']),
        "memory_size": tune.grid_search([512]),
        "module_size": tune.grid_search([None]),
        "nmodules": tune.grid_search([4]),
        "module_task_dim": tune.grid_search([1]),
        "phi_l1_coeff": tune.grid_search([0]),
      },
    ]

  elif search == 'conv_msf2':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([seed]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv4']),
        "memory_size": tune.grid_search([512]),
        "module_size": tune.grid_search([None]),
        "module_task_dim": tune.grid_search([1]),
        "nmodules": tune.grid_search([4]),
        "seperate_value_params": tune.grid_search([True, False]),
        "cumulant_source": tune.grid_search(['conv', 'lstm']),
      } for seed in [1, 2]
    ]

  elif search == 'genv4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([2, 3]),
        "agent": tune.grid_search(['usfa_lstm']),
        "setting": tune.grid_search(['genv4']),
      },
    ]

  elif search == 'multiv8':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
        "setting": tune.grid_search(['multiv8']),
      },
    ]

  else:
    raise NotImplementedError(search)

  return space