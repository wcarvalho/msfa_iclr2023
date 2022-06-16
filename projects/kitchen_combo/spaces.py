from ray import tune

def get(search):
  if search == 'baselines':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
        },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm',]),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          "reward_coeff": tune.grid_search([1, 10, 100]),
        },
    ]
  elif search == 'msf_reward':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          "reward_coeff": tune.grid_search([1, 10, 100]),
        },
    ]
  elif search == 'msf_farm':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf',]),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          "reward_coeff": tune.grid_search([10]),
          "seperate_cumulant_params": tune.grid_search([True, False]),
          "module_attn_heads": tune.grid_search([1, 2]),
        },
    ]


  else:
    raise NotImplementedError(search)

  return space
