from ray import tune

def get(search, agent):
  if search == 'r2d1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([15_000_000]),
          'setting': tune.grid_search(['procgen_easy', 'procgen_easy_medium']),
          "group": tune.grid_search(['r2d1_procgen-2']),
          # 'min_replay_size': tune.grid_search([100, 10_000]),
          # 'importance_sampling_exponent': tune.grid_search([0, .6]),
          # 'out_q_layers': tune.grid_search([1, 2]),
        },
        # {
        #   "seed": tune.grid_search([1]),
        #   "agent": tune.grid_search(['r2d1']),
        #   "max_number_of_steps": tune.grid_search([30_000_000]),
        #   'setting': tune.grid_search(['procgen_easy']),
        #   'min_replay_size': tune.grid_search([100]),
        # }
    ]

  elif search == 'r2d1_procgen_easy':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search([
            'procgen_easy',
            'procgen_easy_medium'
            # 'procgen_easy_hard',
            ]),
          "group": tune.grid_search(['r2d1_procgen-2']),
        }
    ]
  elif search == 'msf_procgen_easy':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['procgen_easy']),
          'reward_coeff': tune.grid_search([1.0, 10.0, 50.0, 100.0]),
          "group": tune.grid_search(['procgen-3']),
        }
    ]
  elif search == 'usfa_procgen_easy':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['procgen_easy']),
          'reward_coeff': tune.grid_search([1.0, 10.0, 50.0, 100.0]),
          "group": tune.grid_search(['procgen-3']),
        }
    ]

  # -----------------------
  # taskgen
  # -----------------------
  elif search == 'r2d1_taskgen_easy':
    space = [
        {
          "seed": tune.grid_search([1, 2]),
          "agent": tune.grid_search(['r2d1']),
          "group": tune.grid_search(['taskgen_easy-9']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['taskgen_long_easy']),
          # 'importance_sampling_exponent': tune.grid_search([0.0, 0.6]),
          # 'r2d1_loss': tune.grid_search(['n_step_q_learning']),
        }
    ]
  elif search == 'r2d1farm_taskgen_easy':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3, 4]),
          "agent": tune.grid_search(['r2d1_farm']),
          "group": tune.grid_search(['r2d1_baseline-1']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['taskgen_long_easy']),
          "farm_policy_task_input": tune.grid_search([False]),
          "farm_task_input": tune.grid_search([True])
       },
        {
          "seed": tune.grid_search([1, 2, 3, 4]),
          "agent": tune.grid_search(['r2d1_farm']),
          "group": tune.grid_search(['r2d1_baseline-1']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['taskgen_long_easy']),
          "farm_policy_task_input": tune.grid_search([True]),
          "farm_task_input": tune.grid_search([False])
       }
    ]

  elif search == 'usfa_taskgen_easy':
    space = [
      {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['taskgen_long_easy']),
          "group": tune.grid_search(['balance-1']),
          "balance_reward": tune.grid_search([.05, .10, .5, 1.0]),
      }
    ]
  elif search == 'msf_taskgen_easy':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
          'setting': tune.grid_search(['taskgen_long_easy']),
          "group": tune.grid_search(['balance-1']),
          "balance_reward": tune.grid_search([.05, .10, .5, 1.0]),
        },
    ]



  elif search == 'one_policy':
    shared = {
      "seed": tune.grid_search([1, 2, 3]),
      "max_number_of_steps": tune.grid_search([10_000_000]),
      'setting': tune.grid_search(['taskgen_long_easy']),
      'npolicies': tune.grid_search([1]),
      'group': tune.grid_search(['one_policy-1']),
    }
    space = [
        {
          "agent": tune.grid_search(['usfa_lstm']),
          "eval_task_support": tune.grid_search(['eval']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "eval_task_support": tune.grid_search(['eval']),
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

    ]

  # ======================================================
  # Final
  # ======================================================
  elif search == 'taskgen_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['taskgen_long_easy']),
      "group": tune.grid_search(['taskgen_final-2']),
      "max_episodes": tune.grid_search([4]),
      "max_number_of_steps": tune.grid_search([10_000_000]),
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

  elif search == 'procgen_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['procgen_easy']),
      "group": tune.grid_search(['procgen_final-2']),
      "label": tune.grid_search(['env_rew_coeff']),
      "max_number_of_steps": tune.grid_search([15_000_000]),
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

  else:
    raise NotImplementedError(search)


  return space
