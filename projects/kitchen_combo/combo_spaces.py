from ray import tune

def get(search, agent):
  if search == 'r2d1_norest':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['small_remove']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['struct_embed']),
          'setting': tune.grid_search(['small_remove']),
        },

    ]

  elif search == 'test_lp':
    space = [
        {
          "seed": tune.grid_search(list(range(1,1+4*5))),
          "agent": tune.grid_search(['r2d1',]),
          "label": tune.grid_search(['v12',]),
          "max_number_of_steps": tune.grid_search([20_000]),
        },

    ]


  elif search == 'test_remove':
    space = [
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['msf']),
        #   "max_number_of_steps": tune.grid_search([2_000_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'setting': tune.grid_search(['test_remove']),
        #   'module_size': tune.grid_search([256]),
        # },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_remove']),
          'module_size': tune.grid_search([128]),
          'sf_net': tune.grid_search(['relational_action']),
          'relate_residual': tune.grid_search(['skip']),
          'relation_position_embed': tune.grid_search([16]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_remove']),
          'module_size': tune.grid_search([128]),
          'sf_net': tune.grid_search(['relational_action']),
          'relate_residual': tune.grid_search(['sigtanh']),
          'relation_position_embed': tune.grid_search([16]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_remove']),
          'module_size': tune.grid_search([128]),
          'sf_net': tune.grid_search(['relational_action']),
          'relate_residual': tune.grid_search(['concat']),
          'relation_position_embed': tune.grid_search([16]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_remove']),
          'module_size': tune.grid_search([128]),
          'sf_net': tune.grid_search(['relational_action']),
          'relate_residual': tune.grid_search(['output']),
          'relation_position_embed': tune.grid_search([16]),
        },
    ]
  elif search == 'test12_relate':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([5_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_remove']),
          'module_size': tune.grid_search([128]),
          'sf_net': tune.grid_search(['relational_action']),
          'npolicies': tune.grid_search([3]),
          'relate_residual': tune.grid_search(['concat']),
          'relation_position_embed': tune.grid_search([16]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([5_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_remove']),
          'module_size': tune.grid_search([128]),
        },
    ]

  elif search == 'test_remove_baselines':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_remove']),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_remove']),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_remove']),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_remove']),
        },

    ]


  # ======================================================
  # Final
  # ======================================================
  elif search == 'test_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3]),
      'setting': tune.grid_search(['test_remove']),
      "group": tune.grid_search(['test-final-2']),
      "max_number_of_steps": tune.grid_search([2_000_000]),
      'task_embedding': tune.grid_search(['none']),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
    ]
  elif search == 'small_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3]),
      'setting': tune.grid_search(['small_remove']),
      "group": tune.grid_search(['small-final-2']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
      'task_embedding': tune.grid_search(['none']),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
    ]
  elif search == 'medium_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['medium_remove']),
      "group": tune.grid_search(['medium-final-2']),
      "max_number_of_steps": tune.grid_search([10_000_000]),
      'task_embedding': tune.grid_search(['none']),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
    ]

  else:
    raise NotImplementedError(search)


  return space
