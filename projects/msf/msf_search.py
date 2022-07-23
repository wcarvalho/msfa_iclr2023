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

  elif search == 'msf_mask':
    space = [
        {
          "group": tune.grid_search(['msf_mask6']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "reward_coeff": tune.grid_search([1]),
          "qaux_mask_loss": tune.grid_search([True]),
          "sf_mask_loss": tune.grid_search([True]),
        },
        {
          "group": tune.grid_search(['msf_mask6']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "reward_coeff": tune.grid_search([1]),
          "qaux_mask_loss": tune.grid_search([False]),
          "sf_mask_loss": tune.grid_search([False]),
        },
        {
          "group": tune.grid_search(['msf_mask6']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "reward_coeff": tune.grid_search([10]),
          "qaux_mask_loss": tune.grid_search([True]),
          "sf_mask_loss": tune.grid_search([True]),
        },
        {
          "group": tune.grid_search(['msf_mask6']),
          "seed": tune.grid_search([1, 2, 3]),
          "agent": tune.grid_search(['msf']),
          "reward_coeff": tune.grid_search([10]),
          "qaux_mask_loss": tune.grid_search([False]),
          "sf_mask_loss": tune.grid_search([False]),
        }
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
      "group": tune.grid_search(['borsa_final-1']),
      "max_number_of_steps": tune.grid_search([5_000_000]),
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

  elif search == 'ablate_shared':
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
  else:
    raise NotImplementedError(search)

  return space
