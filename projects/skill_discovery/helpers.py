import os.path
import yaml

import acme
import functools

from acme import wrappers
import dm_env
import rlax


from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss
from losses import cumulants
from losses import msfa_stats


from projects.msf.helpers import q_aux_sf_loss
from projects.kitchen_gridworld import nets
from projects.kitchen_gridworld import configs

from envs.acme.tasks_wrapper import TrainTasksWrapper
from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor



# ======================================================
# Environment
# ======================================================
def make_environment(evaluation: bool = False,
                     tile_size=8,
                     room_size=6,
                     num_dists=0,
                     max_text_length=10,
                     path='.',
                     task_kinds:str='pickup',
                     ) -> dm_env.Environment:
  """Summary
  
  Args:
      evaluation (bool, optional): Description
      tile_size (int, optional): Description
      room_size (int, optional): Description
      num_dists (int, optional): Description
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

  if task_kinds == 'none':
    raise NotImplementedError

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai_kitchen/vocab.json"))

  all_level_kwargs=dict(
    'reward'='pickup')
  env = BabyAISkills(
    room_size=room_size,
    num_dists=num_dists,
    all_level_kwargs=all_level_kwargs,
    wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size),
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc,
        max_length=max_text_length)],
    )

  # wrappers for dm_env: used by agent/replay buffer
  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(mission='task')),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)

def load_agent_settings(agent, env_spec, config_kwargs=None, max_vocab_size=30):
  default_config = dict(max_vocab_size=max_vocab_size)
  default_config.update(config_kwargs or {})

  if agent == "r2d1":
  # Recurrent DQN (2.2M params)
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.R2D1Config(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_embedding='language',
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    loss_label = 'r2d1'
    eval_network = config.eval_network
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
