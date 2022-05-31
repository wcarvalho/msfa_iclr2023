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

  elif search == 'cov5':
    """
    Next:
    """
    space = [
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "cov_coeff": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5]),
      #   "cov_loss": tune.grid_search(['l2_corr']),
      #   "nmodules": tune.grid_search([4]),
      #   "lang_task_dim": tune.grid_search([16]),
      # },
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "cov_coeff": tune.grid_search([1e-3, 1e-4]),
        "cov_loss": tune.grid_search(['l2_cov']),
        "nmodules": tune.grid_search([4]),
        "lang_task_dim": tune.grid_search([16]),
      },
    ]

  elif search == 'genv3_2':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([3]),
        "agent": tune.grid_search(['r2d1']),
        "setting": tune.grid_search(['genv3']),
      },
    ]

  elif search == 'phi8':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "learning_rate": tune.grid_search([1e-3]),
        "phi_l1_coeff": tune.grid_search([1e-3, 1e-4]),
        "module_l1": tune.grid_search([True, False]),
        "importance_sampling_exponent": tune.grid_search([.6]),
        "nmodules": tune.grid_search([4]),
        "module_task_dim": tune.grid_search([4]),
      },
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "learning_rate": tune.grid_search([5e-3]),
      #   "reward_coeff": tune.grid_search([10]),
      #   "phi_l1_coeff": tune.grid_search([0, 1, .1, .01]),
      #   "nmodules": tune.grid_search([4]),
      #   "module_task_dim": tune.grid_search([4]),
      # },
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "learning_rate": tune.grid_search([1, 1e-3]),
      #   "phi_l1_coeff": tune.grid_search([0]),
      #   "nmodules": tune.grid_search([4]),
      #   "module_task_dim": tune.grid_search([4]),
      # },
    ]

  elif search == 'tanhsig':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "w_l1_coeff": tune.grid_search([1e-3, 1e-4]),
        "lang_tanh": tune.grid_search([True]),
        "task_gate": tune.grid_search(['sigmoid']),
        "importance_sampling_exponent": tune.grid_search([.6]),
        "module_l1": tune.grid_search([True, False]),
      },
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "learning_rate": tune.grid_search([5e-3]),
      #   "reward_coeff": tune.grid_search([10]),
      #   "phi_l1_coeff": tune.grid_search([0, 1, .1, .01]),
      #   "nmodules": tune.grid_search([4]),
      #   "module_task_dim": tune.grid_search([4]),
      # },
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "learning_rate": tune.grid_search([1, 1e-3]),
      #   "phi_l1_coeff": tune.grid_search([0]),
      #   "nmodules": tune.grid_search([4]),
      #   "module_task_dim": tune.grid_search([4]),
      # },
    ]

  elif search == 'faster':
    """
    Next:
    """
    space = [
      # { # 6
      #   "seed": tune.grid_search([1]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['genv3']),
      #   "importance_sampling_exponent": tune.grid_search([.6]),
      #   "learning_rate": tune.grid_search([1e-3]),
      # },
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "importance_sampling_exponent": tune.grid_search([.0]),
        "learning_rate": tune.grid_search([1e-2]),
        "reward_coeff": tune.grid_search([1, .5]),
      },
    ]


  elif search == 'size3':
    """
    Next:
    """
    space = [
      { # 6
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['genv3']),
        "memory_size": tune.grid_search([512]),
        "nmodules": tune.grid_search([16]),
        "module_task_dim": tune.grid_search([1]),
        "cov_coeff": tune.grid_search([None, .1]),
        "module_size": tune.grid_search([64]),
        "embed_position": tune.grid_search([16, 32]),
        "module_attn_heads": tune.grid_search([.5]),
      },
    ]
  else:
    raise NotImplementedError(search)

  return space