import os.path
import yaml

import acme
import functools

from acme import wrappers
import dm_env
import rlax


from utils import ObservationRemapWrapper

from agents import td_agent



from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor
from envs.acme.multitask_generic import MultitaskGeneric
from envs.babyai_kitchen.levelgen import KitchenLevel



# ======================================================
# Environment
# ======================================================
def make_environment(evaluation: bool = False,
                     tile_size=8,
                     room_size=6,
                     max_text_length=10,
                     path='.',
                     setting:str='pickup',
                     ) -> dm_env.Environment:
  """Summary
  
  Args:
      evaluation (bool, optional): Description
      tile_size (int, optional): Description
      room_size (int, optional): Description
      step_penalty (float, optional): Description
      task_reps (str, optional): Description
      max_text_length (int, optional): Description
      path (str, optional): Description
      setting (None, optional): Description
  
  Returns:
      dm_env.Environment: Description
  
  Raises:
      RuntimeError: Description
  """
  """
  TODO: 
  """

  if setting == 'no_reward':
    raise NotImplementedError
    all_level_kwargs=dict(
      env_1_obj=dict(task_kinds='none', num_dists=1, reward_coeff=0.0),
      env_2_obj=dict(task_kinds='none', num_dists=2, reward_coeff=0.0),
      env_3_obj=dict(task_kinds='none', num_dists=3, reward_coeff=0.0),
      env_4_obj=dict(task_kinds='none', num_dists=4, reward_coeff=0.0),
      )

  elif setting =='pickup':
    # dict {env_name : env_kwargs}
    all_level_kwargs=dict(
      # use PickupTask class in envs.babyai_kitchen.tasks
      pickup=dict(task_kinds='pickup'), 
      )
  else:
    raise RuntimeError

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai_kitchen/vocab.json"))

  env = MultitaskGeneric(
    tile_size=tile_size,
    room_size=room_size,
    all_level_kwargs=all_level_kwargs,
    path=path,
    wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size),
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc,
        max_length=max_text_length)],
    LevelCls=KitchenLevel,
    )

  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(mission='task')),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)


def load_agent_settings(agent, env_spec, config_kwargs=None):
  default_config = dict()
  default_config.update(config_kwargs or {})

  if agent == "r2d1":
  # Recurrent DQN (2.2M params)
    config = td_agent.R2D1Config(**default_config)

    from archs.recurrent_q_network import RecurrentQNetwork
    NetworkCls=RecurrentQNetwork
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      rnn_size=512,
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs
