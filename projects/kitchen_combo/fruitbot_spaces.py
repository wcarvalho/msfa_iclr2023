from ray import tune

def get(search, agent):
  if search == 'r2d1':
    space = [
        {
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['r2d1']),
          "max_number_of_steps": tune.grid_search([10_000_000]),
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

  # -----------------------
  # taskgen
  # -----------------------

  elif search == 'uvfa_taskgen':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['taskgen_long_easy']),
      "group": tune.grid_search(['taskgen_final-long-3']),
      "max_episodes": tune.grid_search([4]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_no_task']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_noise']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },

    ]

  elif search == 'sfa_taskgen':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['taskgen_long_easy']),
      "group": tune.grid_search(['taskgen_final-3']),
      "max_episodes": tune.grid_search([4]),
    }
    space = [
        {
          "agent": tune.grid_search(['usfa_lstm']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },

    ]

  elif search == 'uvfa_seedgen':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['procgen_easy']),
      "group": tune.grid_search(['seedgen_final-3']),
    }
    space = [
        # {
        #   "agent": tune.grid_search(['r2d1']),
        #   **shared,
        # },
        {
          "agent": tune.grid_search(['r2d1_noise']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },

    ]

  elif search == 'sfa_seedgen':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['procgen_easy']),
      "group": tune.grid_search(['seedgen_final-3']),
    }
    space = [
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },

    ]

  elif search == 'procgen_msf':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['procgen_easy']),
      "group": tune.grid_search(['procgen_msf-4']),
      "max_number_of_steps": tune.grid_search([15_000_000]),
      "clip_rewards": tune.grid_search([True]),
      "env_reward_coeff": tune.grid_search([1.0]),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          'share_add_zeros': tune.grid_search([False]),
          'env_task_dim':  tune.grid_search([2]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          'share_add_zeros': tune.grid_search([False]),
          'env_task_dim':  tune.grid_search([4]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          'share_add_zeros': tune.grid_search([False]),
          'env_task_dim':  tune.grid_search([2]),
          'value_coeff':  tune.grid_search([1.0]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          'share_add_zeros': tune.grid_search([False]),
          'env_task_dim':  tune.grid_search([4]),
          'value_coeff':  tune.grid_search([1.0]),
          **shared,
        },
    ]
  elif search == 'taskgen_final':
    shared = {
      "seed": tune.grid_search([3, 4]),
      'setting': tune.grid_search(['taskgen_long_easy']),
      "group": tune.grid_search(['taskgen_final-3']),
      "max_episodes": tune.grid_search([4]),
      "max_number_of_steps": tune.grid_search([10_000_000]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          # 'eval_task_support': tune.grid_search(['train']),
          **shared,
        },
    ]

  elif search == 'procgen_final':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['procgen_easy']),
      "group": tune.grid_search(['procgen_final-4']),
      "max_number_of_steps": tune.grid_search([15_000_000]),
      "clip_rewards": tune.grid_search([True]),
      "env_reward_coeff": tune.grid_search([1.0]),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },
        {
          "agent": tune.grid_search(['usfa_lstm']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
          'eval_task_support': tune.grid_search(['eval']),
          **shared,
        },
    ]

  elif search == 'ablations':
    # -----------------------
    # shows that importance of having separate parameters
    # -----------------------
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['taskgen_long_easy']),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
         "module_attn_heads": tune.grid_search([0]),
         "group": tune.grid_search(['ablate-relation-params-1']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "seperate_value_params": tune.grid_search([True]),
         "group": tune.grid_search(['ablate-value-params-1']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "image_attn": tune.grid_search([False]),
         "group": tune.grid_search(['ablate-feature-attention-1']),
          **shared,
        },
    ]
  else:
    raise NotImplementedError(search)


  return space
