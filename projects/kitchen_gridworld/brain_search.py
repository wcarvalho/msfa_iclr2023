from ray import tune

def get(search, agent):
  if search == 'slice4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
        "setting": tune.grid_search(['place_sliced']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]

  elif search == 'pickup2':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['pickup']),
        "r2d1_loss": tune.grid_search(['pickup']),
        "max_number_of_steps": tune.grid_search([4_000_000]),
      },
    ]

  elif search == 'pickup_lang2':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['pickup']),
        "group": tune.grid_search(['pickup2']),
        "out_hidden_size": tune.grid_search([512, 1024]),
        "memory_size": tune.grid_search([512, 1024]),
        # "lang_task_dim": tune.grid_search([0, 16]),
        # "learning_rate": tune.grid_search([1e-4]),
        "max_number_of_steps": tune.grid_search([4_000_000]),
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

  elif search == 'place_sliced':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa_lstm', 'r2d1', 'msf']),
        "setting": tune.grid_search(['place_sliced']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
    ]

  elif search == 'multiv9':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([seed]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['multiv9']),
      }
      for seed in [1,2,3]
    ]

  else:
    raise NotImplementedError(search)

  return space