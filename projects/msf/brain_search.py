from ray import tune

def get(search):
  if search == 'usfa':
    space = {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['usfa']),
    }
  elif search == 'r2d1':
    space = [
      {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['r2d1']),
        "min_replay_size": tune.grid_search([100]),
        "max_number_of_steps": tune.grid_search([4_000_000]),
        "r2d1_loss": tune.grid_search(['n_step_q_learning']),
        },
    ]
  elif search == 'r2d1_no_task':
    space = [
      {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['r2d1_no_task']),
        "min_replay_size": tune.grid_search([100]),
        "max_number_of_steps": tune.grid_search([4_000_000]),
        "r2d1_loss": tune.grid_search(['n_step_q_learning']),
        },
    ]
  elif search == 'usfa_lstm':
    space = [
      {
        "seed": tune.grid_search([1, 2, 3, 4]),
        "agent": tune.grid_search(['usfa_lstm']),
        "min_replay_size": tune.grid_search([100]),
        "max_number_of_steps": tune.grid_search([4_000_000]),
        },
    ]
  elif search == 'q_ablate':
    shared = {
        "agent": tune.grid_search(['msf']),
        "seed": tune.grid_search([3,4]),
        "min_replay_size": tune.grid_search([100]),
      }
    space = [
      {
        **shared,
        "value_coeff": tune.grid_search([.5]),
        "q_aux_anneal": tune.grid_search([0.0]),
      },
      # {
      #   **shared,
      #   "value_coeff": tune.grid_search([10.0]),
      #   "q_aux_anneal": tune.grid_search([50_000]),
      #   "q_aux_end_val": tune.grid_search([1e-3]),
      # },
      # {
      #   **shared,
      #   "value_coeff": tune.grid_search([1.0]),
      #   "q_aux_anneal": tune.grid_search([50_000]),
      # },
      # {
      #   **shared,
      #   "value_coeff": tune.grid_search([10.0]),
      #   "q_aux_anneal": tune.grid_search([50_000]),
      #   "q_aux_end_val": tune.grid_search([0.0]),
      # },
      ]
  elif search == 'gate_ablate':
    shared = {
        "agent": tune.grid_search(['msf']),
        "seed": tune.grid_search([1,2]),
        "min_replay_size": tune.grid_search([100]),
      }
    space = [
      {
        **shared,
        "relate_residual": tune.grid_search(['skip']),
      },
      {
        **shared,
        "relate_residual": tune.grid_search(['concat']),
      },
      # {
      #   **shared,
      #   "relate_residual": tune.grid_search(['sigtanh']),
      # },
      ]
  elif search == 'monolothic':
    shared = {
        "agent": tune.grid_search(['msf']),
        "seed": tune.grid_search([1,2, 3]),
      }
    space = [
      {
        **shared,
        "sf_net": tune.grid_search(['flat']),
        "phi_net": tune.grid_search(['flat']),
      },
      ]
  elif search == 'model_ablate':
    shared = {
        "seed": tune.grid_search([1,2]),
      }
    space = [
      # {
      #   **shared, # delta-model w/ shared params
      #   "contrast_module_coeff": tune.grid_search([0.1]),
      #   "seperate_model_params": tune.grid_search([False]),
      #   "normalize_state": tune.grid_search([False]),
      # },
      {
        **shared, # state-model w/ shared params
        "agent": tune.grid_search(['msf_state_model']),
        "contrast_module_coeff": tune.grid_search([0.1]),
        "seperate_model_params": tune.grid_search([True]),
        "normalize_state": tune.grid_search([False]),
      },
      # {
      #   **shared, # time-model w/ shared params
      #   "agent": tune.grid_search(['msf']),
      #   "contrast_module_coeff": tune.grid_search([0.0]),
      #   "contrast_time_coeff": tune.grid_search([0.1]),
      #   "seperate_model_params": tune.grid_search([False]),
      #   "normalize_state": tune.grid_search([False]),
      # },
      ]
  elif search == 'delta_ablate':
    shared = {
        "agent": tune.grid_search(['msf']),
        "seed": tune.grid_search([1,2]),
      }
    space = [
      {
        **shared,
        "contrast_module_coeff": tune.grid_search([0.0]),
        "cumulant_const": tune.grid_search(['delta_concat']),
      },
      {
        **shared,
        "contrast_module_coeff": tune.grid_search([0.0]),
        "cumulant_const": tune.grid_search(['concat']),
      },
      ]
  elif search == 'relate_ablate':
    shared = {
        "agent": tune.grid_search(['msf']),
        "seed": tune.grid_search([1,2,3]),
      }
    space = [
      {
        **shared,
        "sf_net": tune.grid_search(['independent', 'relational']),
        "module_attn_heads": tune.grid_search([2]),
      },
      ]
  elif search == 'msf':
    shared = {
        "agent": tune.grid_search(['msf']),
        "seed": tune.grid_search([3, 4]),
        # "sf_net" : tune.grid_search(['relational']),
        # "resid_mlp" : tune.grid_search([False]),
        # "sf_net_heads" : tune.grid_search([2]),
        # "sf_net_attn_size" : tune.grid_search([256]),
        # "position_hidden" : tune.grid_search([False]),
        # "max_number_of_steps" : tune.grid_search([4_000_000]),
        # "seperate_cumulant_params" : tune.grid_search([False]),
        "farm_task_input" : tune.grid_search([False]),
        # "phi_net" : tune.grid_search(['independent']),
        "cumulant_hidden_size" : tune.grid_search([256]),
        "cumulant_layers" : tune.grid_search([2]),
      }
    space = [
      {
        **shared,
        # "q_aux_anneal" : tune.grid_search([100_000]),
        # "q_aux_end_val" : tune.grid_search([1e-2, 1e-3]),
        # "value_coeff" : tune.grid_search([10.0]),
      },
      # {
      #   **shared,
      #   "relate_residual" : tune.grid_search(['gru', 'concat']),
      # },
      # {
      #   **shared,
      #   "cumulant_const" : tune.grid_search(['delta_concat']),
      #   "contrast_time_coeff" : tune.grid_search([0.1]),
      #   "contrast_module_coeff" : tune.grid_search([0.1]),
      # },
      ]
  elif search == 'baselines':
    space = {
        "seed": tune.grid_search([1,2,3,4]),
        "agent": tune.grid_search(['r2d1']),
    }


  else:
    raise NotImplementedError(search)

  return space
