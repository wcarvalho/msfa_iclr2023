from ray import tune

def get(search, agent):
  if search == 'main1':
    shared = {
      "seed": tune.grid_search(list(range(1, 11))),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['main']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
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
         "agent": tune.grid_search(['usfa']),
          **shared,
        },
    ]

  elif search == 'main2':
    shared = {
      "seed": tune.grid_search(list(range(1, 11))),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['main']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },
    ]

  elif search == 'ablate_modularity':
    # -----------------------
    # shows the importance of having modular architecture
    # -----------------------
    shared = {
      "seed": tune.grid_search(list(range(1, 11))),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['ablate_modularity']),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
         "sf_net": tune.grid_search(['flat']),
         "phi_net": tune.grid_search(['flat']),
         "memory_size": tune.grid_search([None]),
         "module_size": tune.grid_search([140]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "sf_net": tune.grid_search(['flat']),
         "phi_net": tune.grid_search(['independent']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "sf_net": tune.grid_search(['independent']),
         "phi_net": tune.grid_search(['flat']),
         "memory_size": tune.grid_search([None]),
         "module_size": tune.grid_search([140]),
          **shared,
        },
    ]

  elif search == 'ablate_gpi':
    shared = {
      "seed": tune.grid_search(list(range(1,11))),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['ablate_gpi']),
      'npolicies': tune.grid_search([1]),
      "eval_task_support": tune.grid_search(['eval']),
    }
    space = [
        {
          "agent": tune.grid_search(['usfa_lstm']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          **shared,
        },
        {
         "agent": tune.grid_search(['usfa']),
          **shared,
        },
    ]


  else:
    raise NotImplementedError(search)

  return space
