from ray import tune

def get(search, agent):
  if search == 'r2d1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([50_000_000]),
          'setting': tune.grid_search(['original_easy']),
          'min_replay_size': tune.grid_search([100, 10_000]),
          'importance_sampling_exponent': tune.grid_search([0, .6]),
          # 'out_q_layers': tune.grid_search([1, 2]),
        }
    ]

  elif search == 'r2d1_lr':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([50_000_000]),
          'setting': tune.grid_search(['original_easy']),
          'learning_rate': tune.grid_search([1e-3, 2.5e-4]),
          'r2d1_loss': tune.grid_search(['n_step_q_learning', 'transformed_n_step_q_learning']),
          # 'out_q_layers': tune.grid_search([1, 2]),
        }
    ]
  elif search == 'r2d1_nstep':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([50_000_000]),
          'setting': tune.grid_search(['original_easy']),
          'bootstrap_n': tune.grid_search([5, 40]),
          'trace_length': tune.grid_search([40, 80]),
          # 'out_q_layers': tune.grid_search([1, 2]),
        }
    ]
  elif search == 'r2d1_bs':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([50_000_000]),
          'setting': tune.grid_search(['original_easy']),
          'batch_size': tune.grid_search([64, 128]),
          # 'out_q_layers': tune.grid_search([1, 2]),
        }
    ]
  else:
    raise NotImplementedError(search)


  return space
