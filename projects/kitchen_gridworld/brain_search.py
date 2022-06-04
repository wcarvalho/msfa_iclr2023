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

  elif search == 'similar4':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([2]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
        "setting": tune.grid_search(['similar']),
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
        "module_size": tune.grid_search([128]),
        "memory_size": tune.grid_search([None]),
      },
    ]

  elif search == 'relations':
    """
    Next:
    """
    space = [
      { # 6
        "group": tune.grid_search(['relations']),
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv4']),
        "module_attn_heads": tune.grid_search([.25, .5, .75]),
        "module_size": tune.grid_search([128]),
        "memory_size": tune.grid_search([512]),
      },
    ]

  elif search == 'vocab_fix':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3', 'genv4', 'multiv5', 'multiv7']),
        # "nmodules": tune.grid_search([1, 2, 4, 8]),
        # "module_size": tune.grid_search([128]),
        # "memory_size": tune.grid_search([None]),
      },
    ]

  else:
    raise NotImplementedError(search)

  return space