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
  elif search == 'multi':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['r2d1']),
        "sf_net": tune.grid_search(['relational']),
        "setting": tune.grid_search(['multiv1', 'multiv2', 'multiv3']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      # {
      #   "seed": tune.grid_search([1, 2]),
      #   # "agent": tune.grid_search(['r2d1', 'msf']),
      #   "agent": tune.grid_search(['msf']),
      #   "sf_net": tune.grid_search(['relational']),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      #   "setting": tune.grid_search(['multiv2']),
      #   "lang_task_dim": tune.grid_search([4*6]),
      #   "nmodules": tune.grid_search([6]),
      # },
    ]
  elif search == 'multiv4':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['usfa_lstm']),
        "setting": tune.grid_search(['multiv4']),
        "out_hidden_size": tune.grid_search([512, 1024]),
        "lang_task_dim": tune.grid_search([16, 32]),
       "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['multiv4']),
        "out_hidden_size": tune.grid_search([512, 1024]),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      # {
      #   "seed": tune.grid_search([1, 2]),
      #   # "agent": tune.grid_search(['r2d1', 'msf']),
      #   "agent": tune.grid_search(['msf']),
      #   "sf_net": tune.grid_search(['relational']),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      #   "setting": tune.grid_search(['multiv2']),
      #   "lang_task_dim": tune.grid_search([4*6]),
      #   "nmodules": tune.grid_search([6]),
      # },
    ]

  else:
    raise NotImplementedError(search)

  return space


# -----------------------
# 
# -----------------------