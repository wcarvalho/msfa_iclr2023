from ray import tune

def get(search, agent):
  if search == 'small':
    shared = {
      "seed": tune.grid_search(list(range(1,11))),
      'setting': tune.grid_search(['room_small']),
      "group": tune.grid_search(['small']),
      "max_number_of_steps": tune.grid_search([3_000_000]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },

    ]

  elif search == 'large':
    shared = {
      "seed": tune.grid_search(list(range(1,11))),
      'setting': tune.grid_search(['room_large']),
      "group": tune.grid_search(['large']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
      'num_train_seeds': tune.grid_search([1000]),
      
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          "batch_size": tune.grid_search([128]),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          "batch_size": tune.grid_search([64]),
          **shared,
        },
    ]
  else:
    raise NotImplementedError(search)


  return space
