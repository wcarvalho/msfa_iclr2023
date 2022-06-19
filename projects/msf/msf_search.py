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

  else:
    raise NotImplementedError(search)

  return space
