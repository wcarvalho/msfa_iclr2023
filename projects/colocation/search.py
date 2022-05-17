from ray import tune

def get(search):
  if search == 'r2d1_noise':
    """
    Next:
    """
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['r2d1_noise']),
        "simple": tune.grid_search([False]),
        "nowalls": tune.grid_search([False]),
        "one_room": tune.grid_search([False]),
        "deterministic_rooms":tune.grid_search([True, False]),
        "room_reward": tune.grid_search([.25])
        },
    ]
  elif search=='r2d1_noise_oneseed':
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['r2d1_noise']),
        "simple": tune.grid_search([False]),
        "nowalls": tune.grid_search([False]),
        "one_room": tune.grid_search([False]),
        "deterministic_rooms":tune.grid_search([True, False]),
        "room_reward": tune.grid_search([.25])
        },
    ]


  elif search=='usfa_comparison':
    space = [
      {
        "seed": tune.grid_search([1, 2, 3]),
        "agent": tune.grid_search(['usfa', 'usfa_conv', 'usfa_lstm']),
        "simple": tune.grid_search([False]),
        "nowalls": tune.grid_search([False]),
        "one_room": tune.grid_search([False]),
        "deterministic_rooms":tune.grid_search([True, False]),
        "room_reward": tune.grid_search([.25])
        },
    ]
  elif search=='usfa_comparison_oneseed':
    space = [
      {
        "seed": tune.grid_search([1]),
        "agent": tune.grid_search(['usfa', 'usfa_conv', 'usfa_lstm']),
        "simple": tune.grid_search([False]),
        "nowalls": tune.grid_search([False]),
        "one_room": tune.grid_search([False]),
        "deterministic_rooms":tune.grid_search([True, False]),
        "room_reward": tune.grid_search([.25])
        },
    ]

  else:
    raise NotImplementedError(search)

  return space


# -----------------------
#
# -----------------------