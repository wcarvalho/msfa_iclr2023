from ray import tune

def get(search):
  if search == 'r2d1':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['SmallL2TransferEasy']),
        },
    ]
  elif search == 'usfa_lstm':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['usfa_lstm']),
        "setting": tune.grid_search(['SmallL2TransferEasy', 'SmallL2Transfer', 'SmallL2SliceChill']),
        },
    ]
  
  elif search == 'msf':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['SmallL2TransferEasy']),
      #   "reward_coeff": tune.grid_search([1e-4, 1e-5]),
      #   "contrast_module_coeff": tune.grid_search([0.0]),
      #   "lang_task_dim": tune.grid_search([32, 64]),
      #   "cumulant_const": tune.grid_search(['concat']),
      #   },
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['SmallL2TransferEasy']),
        "reward_coeff": tune.grid_search([1e-4]),
        "contrast_module_coeff": tune.grid_search([0.1]),
        "lang_task_dim": tune.grid_search([32, 64]),
        "cumulant_const": tune.grid_search(['delta_concat']),
        },
    ]

  else:
    raise NotImplementedError(search)

  return space
