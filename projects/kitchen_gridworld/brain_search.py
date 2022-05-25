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

  elif search == 'cov5':
    """
    Next:
    """
    space = [
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "cov_coeff": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5]),
      #   "cov_loss": tune.grid_search(['l2_corr']),
      #   "nmodules": tune.grid_search([4]),
      #   "lang_task_dim": tune.grid_search([16]),
      # },
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "cov_coeff": tune.grid_search([1e-3, 1e-4]),
        "cov_loss": tune.grid_search(['l2_cov']),
        "nmodules": tune.grid_search([4]),
        "lang_task_dim": tune.grid_search([16]),
      },
    ]

  elif search == 'genv3_2':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa_lstm']),
        "label": tune.grid_search(['error_reorder2']),
        "cov_coeff": tune.grid_search([None, 0]),
        "setting": tune.grid_search(['genv3']),
      },
    ]

  elif search == 'phi7':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "learning_rate": tune.grid_search([1e-3]),
        "phi_l1_coeff": tune.grid_search([0, 1, 1e-1, 1e-2]),
        "nmodules": tune.grid_search([6]),
        "module_task_dim": tune.grid_search([4]),
      },
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "learning_rate": tune.grid_search([5e-3]),
      #   "reward_coeff": tune.grid_search([10]),
      #   "phi_l1_coeff": tune.grid_search([0, 1, .1, .01]),
      #   "nmodules": tune.grid_search([4]),
      #   "module_task_dim": tune.grid_search([4]),
      # },
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "learning_rate": tune.grid_search([1, 1e-3]),
      #   "phi_l1_coeff": tune.grid_search([0]),
      #   "nmodules": tune.grid_search([4]),
      #   "module_task_dim": tune.grid_search([4]),
      # },
    ]

  else:
    raise NotImplementedError(search)

  return space
