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

  elif search == 'small_noreset':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search([agent]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'value_coeff': tune.grid_search([0.5]),
          'setting': tune.grid_search(['small_noreset']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search([agent]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'value_coeff': tune.grid_search([0.05]),
          'setting': tune.grid_search(['small_noreset']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search([agent]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'value_coeff': tune.grid_search([0.5]),
          'reward_coeff': tune.grid_search([50]),
          'setting': tune.grid_search(['small_noreset']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search([agent]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'value_coeff': tune.grid_search([0.05]),
          'reward_coeff': tune.grid_search([50]),
          'setting': tune.grid_search(['small_noreset']),
        },
    ]

  elif search == 'test_noreset':
    space = [
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['r2d1']),
        #   "max_number_of_steps": tune.grid_search([1_000_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'setting': tune.grid_search(['test_noreset']),
        # },
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['usfa_lstm']),
        #   "max_number_of_steps": tune.grid_search([1_000_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'setting': tune.grid_search(['test_noreset']),
        # }
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['msf']),
        #   "max_number_of_steps": tune.grid_search([750_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'module_attn_heads': tune.grid_search([2]),
        #   'setting': tune.grid_search(['test_noreset']),
        # },
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['msf']),
        #   "max_number_of_steps": tune.grid_search([750_000]),
        #   'task_embedding': tune.grid_search(['none']),
        #   'out_hidden_size': tune.grid_search([128]),
        #   'policy_layers': tune.grid_search([0]),
        #   'setting': tune.grid_search(['test_noreset']),
        # }
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([750_000]),
          'task_embedding': tune.grid_search(['none']),
          'eval_task_support': tune.grid_search(['train_eval']),
          'setting': tune.grid_search(['test_noreset']),
        },

    ]


  else:
    raise NotImplementedError(search)

  return space
