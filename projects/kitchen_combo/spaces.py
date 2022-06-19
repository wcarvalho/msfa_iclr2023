from ray import tune

def get(search):
  if search == 'r2d12':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['small']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['r2d1',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['struct_embed']),
          'setting': tune.grid_search(['small']),
        },
    ]
  elif search == 'usfa_lstm2':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['small']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'setting': tune.grid_search(['small']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['struct_embed']),
          'setting': tune.grid_search(['small']),
        },
    ]
  elif search == 'msf2':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['none']),
          'setting': tune.grid_search(['small']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['embedding']),
          'setting': tune.grid_search(['small']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'task_embedding': tune.grid_search(['struct_embed']),
          'setting': tune.grid_search(['small']),
        },
    ]


  else:
    raise NotImplementedError(search)

  return space
