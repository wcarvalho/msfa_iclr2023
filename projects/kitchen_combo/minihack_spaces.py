from ray import tune

def get(search, agent):
  if search == 'test_lp':
    space = [
        {
          "seed": tune.grid_search([1, 2]),
          "agent": tune.grid_search(['r2d1']),
          'setting': tune.grid_search(['room_small']),
          'label': tune.grid_search(['v3']),
          "max_number_of_steps": tune.grid_search([20_000]),
          "trace_length": tune.grid_search([80, 40]),
          "discount": tune.grid_search([.99]),
          "batch_size": tune.grid_search([32, 64]),
        }
    ]
  elif search == 'r2d1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          'setting': tune.grid_search(['room_large']),
          'num_train_seeds': tune.grid_search([1000]),
          "group": tune.grid_search(['hps-2']),
          "trace_length": tune.grid_search([80, 40]),
          "discount": tune.grid_search([.99]),
          "batch_size": tune.grid_search([32, 64]),
        }
    ]
  elif search == 'usfa':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['room_large']),
          'num_train_seeds': tune.grid_search([1000]),
          "group": tune.grid_search(['hps-2']),
          "trace_length": tune.grid_search([80, 40]),
          "discount": tune.grid_search([.99]),
          "batch_size": tune.grid_search([32, 64]),
          'eval_task_support': tune.grid_search(['train']),
        }
    ]
  elif search == 'msf':
    space = [
        {
          "seed": tune.grid_search([2]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['room_large']),
          'num_train_seeds': tune.grid_search([1000]),
          "group": tune.grid_search(['hps-2']),
          "trace_length": tune.grid_search([80, 40]),
          "discount": tune.grid_search([.99]),
          "batch_size": tune.grid_search([32, 64]),
          'eval_task_support': tune.grid_search(['train']),
        },
    ]

  # ======================================================
  # Final
  # ======================================================
  elif search == 'small_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['room_small']),
      "group": tune.grid_search(['small_final-2']),
      "max_number_of_steps": tune.grid_search([3_000_000]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
    ]

  elif search == 'large_final_all':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['room_large']),
      "group": tune.grid_search(['large_final-3']),
      'num_train_seeds': tune.grid_search([1000]),
      "max_number_of_steps": tune.grid_search([10_000_000]),
      "batch_size": tune.grid_search([128]),

    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          "memory_size": tune.grid_search([512]),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['train']),
          "memory_size": tune.grid_search([600]), # 3.66M
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          "memory_size": tune.grid_search([480]), # 3.64M
          **shared,
        },
        # {
        #  "agent": tune.grid_search(['msf']),
        #   'eval_task_support': tune.grid_search(['eval']),
        #   "memory_size": tune.grid_search([480]),
        #   **shared,
        # },
    ]
  elif search == 'large_final_2':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['room_large']),
      "group": tune.grid_search(['large_final-2']),
      'num_train_seeds': tune.grid_search([1000]),
      "max_number_of_steps": tune.grid_search([10_000_000]),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          "memory_size": tune.grid_search([480]), # 3.64M
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          "memory_size": tune.grid_search([480]),
          **shared,
        },
    ]
  else:
    raise NotImplementedError(search)


  return space
