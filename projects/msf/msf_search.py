from ray import tune

def get(search):
  if search == 'usfa':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['usfa']),
        "group": tune.grid_search(['baselines5'])
    }
  elif search == 'r2d1':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['r2d1']),
        "group": tune.grid_search(['baselines5'])
    }
  elif search == 'baselines':
    space = {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'msf', 'usfa_lstm', 'usfa']),
    }
  elif search == 'reward':
    space = [
        # {
        #     "seed": tune.grid_search([1]),
        #     "agent": tune.grid_search(['usfa_lstm']),
        #     "reward_coeff": tune.grid_search([10]),
        # },
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "reward_coeff": tune.grid_search([100, 10]),
        }
        ]

  elif search == 'msf_mask':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "reward_coeff": tune.grid_search([10]),
          "qaux_mask_loss": tune.grid_search([True, False]),
          "sf_mask_loss": tune.grid_search([True, False]),
        }
        ]

  else:
    raise NotImplementedError(search)

  return space
