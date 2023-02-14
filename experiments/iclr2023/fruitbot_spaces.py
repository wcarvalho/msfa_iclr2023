from ray import tune

def get(search, agent):
  if search == 'main':
    shared = {
      "seed": tune.grid_search(list(range(1,11))),
      'setting': tune.grid_search(['taskgen_long_easy']),
      "group": tune.grid_search(['main']),
      "max_number_of_steps": tune.grid_search([10_000_000]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1_farm']),
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
         "agent": tune.grid_search(['r2d1']),
          **shared,
        },
    ]
  else:
    raise NotImplementedError(search)


  return space
