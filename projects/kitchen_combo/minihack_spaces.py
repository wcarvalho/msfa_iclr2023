from ray import tune

def get(search, agent):
  if search == 'r2d1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "group": tune.grid_search(['exp-1']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'setting': tune.grid_search(['room_small', 'room_large']),
          'num_train_seeds': tune.grid_search([200, 500]),
          # 'importance_sampling_exponent': tune.grid_search([0.0, 0.6]),
          # 'r2d1_loss': tune.grid_search(['n_step_q_learning']),
        }
    ]
  elif search == 'usfa':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'setting': tune.grid_search(['room_small', 'room_large']),
          'num_train_seeds': tune.grid_search([200, 500]),
          "group": tune.grid_search(['exp-1']),
          'value_coeff': tune.grid_search([.5]),
          'reward_coeff': tune.grid_search([10.0]),
          'eval_task_support': tune.grid_search(['train']),
        }
    ]
  elif search == 'msf':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([20_000_000]),
          'setting': tune.grid_search(['room_small', 'room_large']),
          'num_train_seeds': tune.grid_search([200, 500]),
          "group": tune.grid_search(['exp-1']),
          'reward_coeff': tune.grid_search([10.0]),
          # 'priority_use_aux': tune.grid_search([True, False]),
          # 'priority_weights_aux': tune.grid_search([True, False]),
          # 'max_episodes': tune.grid_search([1.0]),
          'eval_task_support': tune.grid_search(['train']),
        },
    ]
  elif search == 'r2d1_rate':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([2_000_000]),
          'setting': tune.grid_search(['room_small']),
          'num_train_seeds': tune.grid_search([500]),
          "group": tune.grid_search(['rate-limiter-linear']),
          'samples_per_insert': tune.grid_search([6.0, 8.0]),
          # 'priority_use_aux': tune.grid_search([True, False]),
          # 'priority_weights_aux': tune.grid_search([True, False]),
          # 'max_episodes': tune.grid_search([1.0]),
          'eval_task_support': tune.grid_search(['train']),
        },
    ]
  elif search == 'msf_rate':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([2_000_000]),
          'setting': tune.grid_search(['room_small']),
          'num_train_seeds': tune.grid_search([500]),
          "group": tune.grid_search(['rate-limiter-linear']),
          'reward_coeff': tune.grid_search([10.0]),
          'samples_per_insert': tune.grid_search([4.0, 6.0, 8.0]),
          # 'priority_use_aux': tune.grid_search([True, False]),
          # 'priority_weights_aux': tune.grid_search([True, False]),
          # 'max_episodes': tune.grid_search([1.0]),
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
      "group": tune.grid_search(['small_final-1']),
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

  elif search == 'large_final-1':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['room_large']),
      "group": tune.grid_search(['large_final-1']),
      "max_number_of_steps": tune.grid_search([10_000_000]),
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
          "memory_size": tune.grid_search([600]),
          **shared,
        },
    ]
  elif search == 'large_final-2':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['room_large']),
      "group": tune.grid_search(['large_final-1']),
      "max_number_of_steps": tune.grid_search([10_000_000]),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['train']),
          "memory_size": tune.grid_search([480]),
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
