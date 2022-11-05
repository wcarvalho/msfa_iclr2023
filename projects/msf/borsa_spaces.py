from ray import tune

def get(search):
  if search == 'usfa':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3, 4]),
          "agent": tune.grid_search(['usfa']),
          "group": tune.grid_search(['borsa_final-4']),
          "eval_task_support": tune.grid_search(['eval']),
        },
      ]
  elif search == 'r2d1':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['r2d1']),
        "group": tune.grid_search(['baselines5'])
    }
  elif search == 'r2d1_farm':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['r2d1_farm']),
        "group": tune.grid_search(['borsa_final-4']),
        "farm_policy_task_input": tune.grid_search([False]),
        "farm_task_input": tune.grid_search([True])
    }

  elif search == 'usfa_lstm':
    space = [
        {
          "group": tune.grid_search(['balance-1']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "reward_coeff": tune.grid_search([1]),
          "eval_task_support": tune.grid_search(['eval']),
          "balance_reward": tune.grid_search([.05, .50, .75, 1.0]),
        },
    ]
  elif search == 'msf':
    space = [
        {
          "group": tune.grid_search(['value-1']),
          "seed": tune.grid_search([1]),
          "agent": tune.grid_search(['msf']),
          "eval_task_support": tune.grid_search(['eval']),
          "value_coeff": tune.grid_search([.1, .05]),
          "max_number_of_steps": tune.grid_search([5_000_000]),
        },
        ]


  elif search == 'small_noise_gpi_y':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['small_noise-1']),
      "variance": tune.grid_search([.1]),
      "group": tune.grid_search(['small_noise-xl_respawn-1']),
      "label": tune.grid_search(['fix']),
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

  elif search == 'small_noise_gpi_n':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['small_noise-xl_respawn-1']),
      "variance": tune.grid_search([.1]),
      "eval_task_support": tune.grid_search(['eval']),
      "label": tune.grid_search(['fix']),
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

  elif search == 'one_policy_gpi_y':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['one_policy-xl_respawn-1']),
      'npolicies': tune.grid_search([1]),
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

  elif search == 'one_policy_gpi_n':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xxl_nopickup']),
      "group": tune.grid_search(['one_policy-xxl_nopickup-1']),
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

  elif search == 'one_policy_gpi_n2':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['one_policy-xl_respawn-1']),
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

  elif search == 'xl_respawn':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['xl_respawn-3']),
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

  elif search == 'xxl_nopickup':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xxl_nopickup']),
      "group": tune.grid_search(['xxl_nopickup-3']),
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

  elif search == 'gate_search':
    shared = {
      "group": tune.grid_search(['gate_search-4']),
      "seed": tune.grid_search([1, 2, 3, 4]),
      "agent": tune.grid_search(['msf']),
    }
    space = [
        {
          "agent": tune.grid_search(['msf']),
          "share_residual": tune.grid_search(['concat']),
          "memory_size": tune.grid_search([460]),
          'setting': tune.grid_search(['xxl_nopickup']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          "share_residual": tune.grid_search(['concat']),
          "memory_size": tune.grid_search([460]),
          'setting': tune.grid_search(['xl_respawn']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          "share_residual": tune.grid_search(['sigtanh']),
          'setting': tune.grid_search(['xxl_nopickup']),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "share_residual": tune.grid_search(['sigtanh']),
          'setting': tune.grid_search(['xl_respawn']),
          **shared,
        },
    ]

  elif search == 'r2d1_baselines':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['xl_respawn-3']),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1_no_task']),
          **shared,
        },
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },
    ]


  elif search == 'baselines':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['xl_respawn-4']),
    }
    space = [
        # {
        #   "agent": tune.grid_search(['r2d1_no_task']),
        #   **shared,
        # },
        # {
        #   "agent": tune.grid_search(['r2d1_farm']),
        #   **shared,
        # },
        {
          "agent": tune.grid_search(['r2d1_noise']),
          **shared,
        },
        {
          "agent": tune.grid_search(['msf']),
          "share_residual": tune.grid_search(['sigtanh']),
          "memory_size": tune.grid_search([512]),
          "eval_task_support": tune.grid_search(['eval']),
          **shared,
        },
    ]

  # ======================================================
  # Final
  # ======================================================
  elif search == 'replication_1':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['borsa_final-6']),
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

  elif search == 'replication_2':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['borsa_final-6']),
    }
    space = [
        {
          "agent": tune.grid_search(['r2d1_farm']),
          **shared,
        },
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
         "agent": tune.grid_search(['usfa']),
         "eval_task_support": tune.grid_search(['eval']),
          **shared,
        },
    ]
  elif search == 'ablate_modularity':
    # -----------------------
    # shows that importance of having modular architecture
    # -----------------------
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['ablate_modularity-2']),
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
        {
         "agent": tune.grid_search(['msf']),
         "sf_net": tune.grid_search(['independent']),
         "phi_net": tune.grid_search(['independent']),
          **shared,
        },
    ]

  elif search == 'ablate_shared_phi_psi':
    # -----------------------
    # shows that importance of having separate parameters
    # -----------------------
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "group": tune.grid_search(['ablate_shared-2']),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
         "seperate_cumulant_params": tune.grid_search([True]),
         "seperate_value_params": tune.grid_search([True]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "seperate_cumulant_params": tune.grid_search([True]),
         "seperate_value_params": tune.grid_search([False]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "seperate_cumulant_params": tune.grid_search([False]),
         "seperate_value_params": tune.grid_search([True]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "seperate_cumulant_params": tune.grid_search([False]),
         "seperate_value_params": tune.grid_search([False]),
          **shared,
        },
    ]

  elif search == 'ablate_share_attention':
    # -----------------------
    # shows that importance of having separate parameters
    # -----------------------
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['xl_respawn']),
      "schedule_end": tune.grid_search([80e3]),
      "final_lr_scale": tune.grid_search([1e-2]),
      "group": tune.grid_search(['ablate_share_attention-1']),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
         "seperate_value_params": tune.grid_search([True]),
          **shared,
        },
        {
         "agent": tune.grid_search(['msf']),
         "image_attn": tune.grid_search([False]),
          **shared,
        },
    ]
  else:
    raise NotImplementedError(search)

  return space
