from ray import tune

def get(search):
  if search == 'usfa':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['usfa']),
    }
  elif search == 'r2d1':
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1']),
        "min_replay_size": tune.grid_search([100]),
        "out_hidden_size": tune.grid_search([128, 512]),
        "memory_size": tune.grid_search([512, 1024]),
        },
    ]

  else:
    raise NotImplementedError(search)

  return space
