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
  elif search == 'usfa_lstm':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['usfa_lstm']),
        "setting": tune.grid_search(['SmallL2TransferEasy', 'SmallL2Transfer', 'SmallL2SliceChill']),
        },
    ]
  elif search == 'msf':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['SmallL2TransferEasy']),
        "contrast_module_coeff": tune.grid_search([0.0]),
        "lang_task_dim": tune.grid_search([16, 32]),
        "cumulant_const": tune.grid_search(['concat']),
        "cumulant_layers": tune.grid_search([1]),
        },
    ]
  elif search == 'baselines':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm', 'msf']),
        "setting": tune.grid_search(['L2_Multi']),
      },
    ]
  elif search == 'ssf':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2]),
        "agent": tune.grid_search(['msf']),
        "setting": tune.grid_search(['L2_Multi']),
        "sf_net": tune.grid_search(['independent', 'relational']),
        "module_attn_heads": tune.grid_search([0]),
      },
      # {
      #   "seed": tune.grid_search([1, 2]),
      #   "agent": tune.grid_search(['msf']),
      #   "setting": tune.grid_search(['L2_Multi']),
      #   "sf_net": tune.grid_search(['independent']),
      #   "module_attn_heads": tune.grid_search([2]),
      # },
    ]
  elif search == 'monolithic':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([3, 4]),
        "agent": tune.grid_search(['msf_monolithic']),
        "setting": tune.grid_search(['L2_Args_Multi']),
      },
    ]
  elif search == 'big_room_lesslang':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm', 'msf']),
        "setting": tune.grid_search(['L2_Multi']),
        "room_size": tune.grid_search([7]),
      },
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm', 'msf']),
        "setting": tune.grid_search(['L2_Multi']),
        "task_reps": tune.grid_search(['lesslang']),
      },
    ]
  elif search == 'moredists':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1', 'usfa_lstm', 'msf']),
        "setting": tune.grid_search(['L2_Multi']),
        "num_dists": tune.grid_search([5]),
      },
    ]
  else:
    raise NotImplementedError(search)

  return space
