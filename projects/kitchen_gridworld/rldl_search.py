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
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['SmallL2TransferEasy']),
        "contrast_module_coeff": tune.grid_search([0.0]),
        "lang_task_dim": tune.grid_search([16, 32]),
        "cumulant_const": tune.grid_search(['concat']),
        "cumulant_layers": tune.grid_search([1]),
        },
    ]

  elif search == 'msf_modules':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1, 2]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['SmallL2TransferEasy']),
      #   "out_hidden_size": tune.grid_search([128]),
      #   "nmodules": tune.grid_search([4]),
      #   "lang_task_dim": tune.grid_search([16]),
      #   },
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['SmallL2TransferEasy']),
        "out_hidden_size": tune.grid_search([512]),
        "nmodules": tune.grid_search([8]),
        "lang_task_dim": tune.grid_search([16, 32]),
        },
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['SmallL2TransferEasy']),
        "out_hidden_size": tune.grid_search([1024]),
        "nmodules": tune.grid_search([8]),
        "lang_task_dim": tune.grid_search([32]),
        },
    ]
  else:
    raise NotImplementedError(search)

  return space
