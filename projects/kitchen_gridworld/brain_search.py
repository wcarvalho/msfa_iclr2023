from ray import tune

def get(search):
  if search == 'r2d1':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['L2_Multi']),
        },
    ]
  elif search == 'multi':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['multiv5']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['r2d1', 'usfa_lstm']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]
  elif search == 'clean':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1, 2, 3]),
      #   "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
      #   "max_number_of_steps": tune.grid_search([10_000_000]),
      #   "setting": tune.grid_search(['clean']),
      # },
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['msf']),
        "max_number_of_steps": tune.grid_search([10_000_000]),
        "setting": tune.grid_search(['clean']),
      },
    ]

  else:
    raise NotImplementedError(search)

  return space


# -----------------------
# 
# -----------------------