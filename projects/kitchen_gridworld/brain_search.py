from ray import tune

def get(search):
  if search == 'r2d1':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['L2_Multi']),
        },
    ]
  elif search == 'multi':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1, 2, 3]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['multiv5']),
      #   "room_size": tune.grid_search([7]),
      #   "num_dists": tune.grid_search([5]),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      # },
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
        "setting": tune.grid_search(['multiv5']),
        "room_size": tune.grid_search([7]),
        "num_dists": tune.grid_search([0]),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]
  elif search == 'params':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['multiv5']),
        "nmodules": tune.grid_search([4]),
        "seperate_value_params": tune.grid_search([True, False]),
        "seperate_cumulant_params": tune.grid_search([True, False]),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]
  elif search == 'layernorm':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['multiv1']),
        "sf_layernorm": tune.grid_search(['sf_input', 'sf']),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      # {
      #   "seed": tune.grid_search([1, 2, 3]),
      #   "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
      #   "setting": tune.grid_search(['multiv5']),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      # },
    ]
  elif search == 'reward':
    """
    Next:
    """
    space = [
      # {
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf', 'usfa_lstm']),
      #   "setting": tune.grid_search(['multiv5']),
      #   "reward_coeff": tune.grid_search([10, 1, 1e-1]),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      # },
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf', 'usfa_lstm']),
        "setting": tune.grid_search(['multiv5']),
        "module_l1": tune.grid_search([True]),
        "reward_coeff": tune.grid_search([10, 1, 1e-1]),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
    ]
  elif search == 'reward_msf':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['multiv5']),
        "reward_coeff": tune.grid_search([100, 50, 10, 5]),
        "phi_l1_coeff": tune.grid_search([0.0]),
        "lang_task_dim": tune.grid_search([16]),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      # {
      #   "seed": tune.grid_search([1, 2, 3]),
      #   "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
      #   "setting": tune.grid_search(['multiv5']),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      # },
    ]
  elif search == 'phi':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['multiv5']),
        "phi_l1_coeff": tune.grid_search([.01]),
        "module_l1": tune.grid_search([True, False]),
        "lang_task_dim": tune.grid_search([128]),
        "max_number_of_steps": tune.grid_search([30_000_000]),
      },
      # {
      #   "seed": tune.grid_search([1, 2, 3]),
      #   "agent": tune.grid_search(['r2d1', 'usfa_lstm']),
      #   "setting": tune.grid_search(['multiv5']),
      #   "max_number_of_steps": tune.grid_search([30_000_000]),
      # },
    ]

  else:
    raise NotImplementedError(search)

  return space


# -----------------------
# 
# -----------------------