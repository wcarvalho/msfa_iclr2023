from ray import tune

def get(search):
  if search == 'r2d1':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
        },
    ]
  elif search == 'usfa_lstm':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'value_coeff': tune.grid_search([.05]),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'value_coeff': tune.grid_search([.05]),
        },
    ]
  elif search == 'test2':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          'task_embedding': tune.grid_search(['embedding']),
          "max_number_of_steps": tune.grid_search([1_00_000]),
          "setting": tune.grid_search(['test']),
          "out_hidden_size": tune.grid_search([s]),
        } for s in [128, 512]
    ] + [
            {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          'task_embedding': tune.grid_search(['embedding']),
          "max_number_of_steps": tune.grid_search([1_00_000]),
          "setting": tune.grid_search(['test']),
          "out_hidden_size": tune.grid_search([s]),
        } for s in [128, 512]
    ]
  elif search == 'msf':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'value_coeff': tune.grid_search([.05]),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'value_coeff': tune.grid_search([.05]),
        },
    ]

  elif search == 'msf_no_mask':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'qaux_mask_loss': tune.grid_search([False]),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'qaux_mask_loss': tune.grid_search([False]),
        },
    ]

  else:
    raise NotImplementedError(search)

  return space
