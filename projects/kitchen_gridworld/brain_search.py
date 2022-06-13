from ray import tune

def get(search, agent=''):
  agent = agent or 'r2d1'
  if search == 'slice5':
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['place_sliced']),
        "max_number_of_steps": tune.grid_search([40_000_000]),
      },
    ]

  elif search == 'gen5_modr2d1':
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "setting": tune.grid_search(['genv5']),
        "agent": tune.grid_search(['modr2d1']),
        "struct_w": tune.grid_search([False, True]),
        "nmodules": tune.grid_search([4]),
        "task_reps": tune.grid_search(['object']),
        "dot_qheads": tune.grid_search([False, True]),
        "max_number_of_steps": tune.grid_search([10_000_000]),
      }
    ]
  elif search == 'gen5':
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "setting": tune.grid_search(['genv5']),
        "agent": tune.grid_search(['r2d1', 'msf']),
        "task_reps": tune.grid_search(['no_object']),
        "max_number_of_steps": tune.grid_search([10_000_000]),
      }
    ]

  elif search == 'cook5':
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['cook']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]

  elif search == 'similar5':
    space = [
      { # 6
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['similar']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]

  elif search == 'pickup6':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search([agent]),
        "setting": tune.grid_search(['pickup']),
        "symbolic": tune.grid_search([False, True]),
        "label": tune.grid_search(['ray_more_cpu']),
        "max_number_of_steps": tune.grid_search([2_000_000]),
      },
    ]

  elif search == 'modr2d1':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['modr2d1']),
        "setting": tune.grid_search(['pickup']),
        "struct_w": tune.grid_search([False, True]),
        "nmodules": tune.grid_search([4]),
        "dot_qheads": tune.grid_search([False, True]),
        "max_number_of_steps": tune.grid_search([2_000_000]),
      },
    ]

  elif search == 'pickup_lang3':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search([
          'r2d1']),
        # "word_initializer": tune.grid_search(['RandomNormal', 'TruncatedNormal']),
        # "samples_per_insert": tune.grid_search([0.0, 6.0]),
        "sequence_period": tune.grid_search([40]),
        # "burn_period": tune.grid_search([40]),
        "setting": tune.grid_search(['pickup']),
        "mask_loss": tune.grid_search([True, False]),
        "max_number_of_steps": tune.grid_search([4_000_000]),
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


  else:
    raise NotImplementedError(search)

  return space