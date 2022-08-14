from ray import tune

def get(search):
  if search == 'usfa':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa']),
          "group": tune.grid_search(['baselines5']),
          "eval_task_support": tune.grid_search(['train_eval']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa']),
          "group": tune.grid_search(['baselines5']),
          "eval_task_support": tune.grid_search(['eval']),
        },
      ]
  elif search == 'r2d1':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['r2d1']),
        "group": tune.grid_search(['baselines5'])
    }

  elif search == 'usfa_lstm':
    space = [
        {
          "group": tune.grid_search(['usfa_lstm_mask5']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "reward_coeff": tune.grid_search([1]),
          "qaux_mask_loss": tune.grid_search([True]),
          "sf_mask_loss": tune.grid_search([True]),
        },
        {
          "group": tune.grid_search(['usfa_lstm_mask5']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "reward_coeff": tune.grid_search([1]),
          "qaux_mask_loss": tune.grid_search([False]),
          "sf_mask_loss": tune.grid_search([False]),
        },
        {
          "group": tune.grid_search(['usfa_lstm_mask5']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "reward_coeff": tune.grid_search([10]),
          "qaux_mask_loss": tune.grid_search([True]),
          "sf_mask_loss": tune.grid_search([True]),
        },
        {
          "group": tune.grid_search(['usfa_lstm_mask5']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['usfa_lstm']),
          "reward_coeff": tune.grid_search([10]),
          "qaux_mask_loss": tune.grid_search([False]),
          "sf_mask_loss": tune.grid_search([False]),
        }
    ]
  elif search == 'msf':
    space = [
        {
          "group": tune.grid_search(['msf-lr-3']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "schedule_end": tune.grid_search([40e3, 80e3]),
          "final_lr_scale": tune.grid_search([1e-1, 1e-2]),
          # "variable_update_period": tune.grid_search([600, 800, 1000, 1200]),
          # "learning_rate": tune.grid_search([1e-3, 5e-4, 1e-4, 5e-4]),
          # "struct_policy_input": tune.grid_search([False]),
          # "sf_share_output": tune.grid_search([True]),
          # "phi_mask_loss": tune.grid_search([False, True]),
          # "elemwise_sf_loss": tune.grid_search([False]),
          # "elemwise_phi_loss": tune.grid_search([False]),
          # "samples_per_insert": tune.grid_search([0.0, 10.0]),
        },
        ]

  elif search == 'msf_struct':
    space = [
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "struct_policy_input": tune.grid_search([True]),
          "eval_task_support": tune.grid_search(['train_eval']),
          "group": tune.grid_search(['msf_struct6']),

        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "struct_policy_input": tune.grid_search([True]),
          "eval_task_support": tune.grid_search(['train']),
          "group": tune.grid_search(['msf_struct6']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "struct_policy_input": tune.grid_search([False]),
          "eval_task_support": tune.grid_search(['train_eval']),
          "group": tune.grid_search(['msf_struct6']),
        },
        {
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "struct_policy_input": tune.grid_search([False]),
          "eval_task_support": tune.grid_search(['train']),
          "group": tune.grid_search(['msf_struct6']),
        }
    ]
  # ======================================================
  # Final
  # ======================================================
  elif search == 'replication':
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['large_respawn']),
      "group": tune.grid_search(['borsa_final-3']),
      "memory_size": tune.grid_search([512]),
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

  elif search == 'ablate_modularity':
    # -----------------------
    # shows that importance of having modular architecture
    # -----------------------
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['large_respawn']),
      "group": tune.grid_search(['ablate_modularity-1']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
         "sf_net": tune.grid_search(['flat']),
         "phi_net": tune.grid_search(['flat']),
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
      'setting': tune.grid_search(['large_respawn']),
      "group": tune.grid_search(['ablate_shared-1']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
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
      'setting': tune.grid_search(['large_respawn']),
      "group": tune.grid_search(['ablate_share_attention-1']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
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

  elif search == 'ablate_relation_heads':
    # -----------------------
    # shows that importance of having separate parameters
    # -----------------------
    shared = {
      "seed": tune.grid_search([1, 2, 3, 4]),
      'setting': tune.grid_search(['large_respawn']),
      "group": tune.grid_search(['ablate_share_attention-1']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
    }
    space = [
        {
         "agent": tune.grid_search(['msf']),
         "module_attn_heads": tune.grid_search([n]),
          **shared,
        } for n in [0, 1, 3, 4]
    ]

  else:
    raise NotImplementedError(search)

  return space
