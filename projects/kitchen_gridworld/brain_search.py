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

  elif search == 'genv4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
        "setting": tune.grid_search(['genv4']),
      },
    ]
  elif search == 'size4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv4']),
        "nmodules": tune.grid_search([1, 2, 4, 8]),
        "module_size": tune.grid_search([32]),
      },
    ]








  else:
    raise NotImplementedError(search)

  return space
