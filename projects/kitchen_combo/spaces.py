from ray import tune

def get(search, agent):
  if search == 'r2d1_norest':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['small_noreset']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['struct_embed']),
          'setting': tune.grid_search(['small_noreset']),
        },

    ]

  elif search == 'test_noreset':
    space = [
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['msf']),
        #   "max_number_of_steps": tune.grid_search([2_000_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'setting': tune.grid_search(['test_noreset']),
        #   'module_size': tune.grid_search([256]),
        # },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_noreset']),
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
          'setting': tune.grid_search(['test_noreset']),
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
          'setting': tune.grid_search(['test_noreset']),
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
          'setting': tune.grid_search(['test_noreset']),
          'module_size': tune.grid_search([128]),
          'sf_net': tune.grid_search(['relational_action']),
          'relate_residual': tune.grid_search(['output']),
          'relation_position_embed': tune.grid_search([16]),
        },
    ]
  elif search == 'test12_skil':
    space = [
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['msf']),
        #   "max_number_of_steps": tune.grid_search([2_000_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'setting': tune.grid_search(['test_noreset']),
        #   'module_size': tune.grid_search([256]),
        # },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([5_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_noreset']),
          'module_size': tune.grid_search([64]),
          'sf_net': tune.grid_search(['relational_action']),
          'npolicies': tune.grid_search([3]),
          'relate_residual': tune.grid_search(['skip']),
          'relation_position_embed': tune.grid_search([16]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([5_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_noreset']),
          'module_size': tune.grid_search([64]),
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
          'setting': tune.grid_search(['test_noreset']),
          'module_size': tune.grid_search([32]),
          'sf_net': tune.grid_search(['relational_action']),
          'npolicies': tune.grid_search([3]),
          'relate_residual': tune.grid_search(['skip']),
          'relation_position_embed': tune.grid_search([16]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([5_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['test_noreset']),
          'module_size': tune.grid_search([32]),
          'sf_net': tune.grid_search(['relational_action']),
          'npolicies': tune.grid_search([3]),
          'relate_residual': tune.grid_search(['concat']),
          'relation_position_embed': tune.grid_search([16]),
        },
    ]

  elif search == 'test_noreset_baselines':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_noreset']),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_noreset']),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_noreset']),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([3_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'label': tune.grid_search(['redo']),
          'setting': tune.grid_search(['test_noreset']),
        },

    ]


  else:
    raise NotImplementedError(search)

  return space
