import acme
import functools

from acme import wrappers
import dm_env
import rlax

from typing import NamedTuple

from acme import types
from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses import usfa as usfa_losses
from losses.vae import VaeAuxLoss
from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss
from losses import cumulants
from modules.ensembles import QLearningEnsembleLoss

from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper
from envs.acme.multitask_generic import MultitaskGeneric
from envs.babyai_kitchen.kitchen_combo_level import KitchenComboLevel


from projects.kitchen_gridworld import nets
from projects.kitchen_gridworld import helpers as kitchen_helpers
from projects.msf import helpers as msf_helpers
from projects.kitchen_combo import configs

class ComboObsTuple(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest
  train_tasks: types.Nest

def make_environment(evaluation: bool = False,
                     tile_size=8,
                     setting='medium',
                     path='.',
                     task2rew=None,
                     ) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
      tile_size (int, optional): number of pixels per grid-cell.
      setting (str, optional): `small` env, `medium` env or `large` env.
      path (str, optional): path in system where running from.
  
  Returns:
      dm_env.Environment: Multitask GotoAvoid environment is returned.
  """
  settings = dict(
    test=dict(room_size=5, ntasks=1),
    small=dict(room_size=7, ntasks=1),
    medium=dict(room_size=9, ntasks=2),
  )
  if task2rew is None:
    train = {
          "0.slice":{
              "slice" : 1,
              "chill" : 0,
              "clean" : 0,
              },
          "1.chill":{
              "slice" : 0,
              "chill" : 1,
              "clean" : 0,
              },
          "2.clean":{
              "slice" : 0,
              "chill" : 0,
              "clean" : 1,
              }
      }
    if evaluation:
      task2rew={
          **train,
          "3.slice-chill":{
              "slice" : 1,
              "chill" : 1,
              "clean" : 0,
              },
          "4.slice-clean":{
              "slice" : 1,
              "chill" : 0,
              "clean" : 1,
              },
          "5.chill-clean":{
              "slice" : 0,
              "chill" : 1,
              "clean" : 1,
              },
          "6.slice-chill-clean":{
              "slice" : 1,
              "chill" : 1,
              "clean" : 1,
              },
          "7.slice-neg-chill-clean":{
              "slice" : 1,
              "chill" : -1,
              "clean" : 1,
              },
          "8.slice-neg-chill-neg-clean":{
              "slice" : 1,
              "chill" : -1,
              "clean" : -1,
              },
      }
    else:
      task2rew=train

  if setting == 'test':
    task2rew=dict(pickup={"pickup" : 1})

  all_level_kwargs=dict()
  for key, item in task2rew.items():
    all_level_kwargs[key] = dict(
        task2reward=item,
      )

  env = MultitaskGeneric(
    tile_size=tile_size,
    all_level_kwargs=all_level_kwargs,
    ObsTuple=ComboObsTuple,
    path=path,
    wrappers=[functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size)],
    LevelCls=KitchenComboLevel,
    **settings[setting],
    )

  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(
            mission='task',
            )),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)



def load_agent_settings(agent, env_spec, config_kwargs=None):
  default_config = dict()
  default_config.update(config_kwargs or {})
  agent = agent.lower()

  if agent == "r2d1":
  # Recurrent DQN/UVFA
    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config()],
      dict_configs=default_config
    )

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_embedding=config.task_embedding,
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning

    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.USFAConfig(),
        configs.QAuxConfig(),
        configs.RewardConfig()],
      dict_configs=default_config
      )

    NetworkCls=nets.usfa # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_embedding=config.task_embedding,
      use_separate_eval=True,
      predict_cumulants=True,)

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.sf_loss,
      mask_loss=config.sf_mask_loss,
      shorten_data_for_cumulant=True,
      extract_cumulants=losses.cumulants_from_preds,
      aux_tasks=[
        msf_helpers.q_aux_sf_loss(config),
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,
          loss=config.reward_loss,
          balance=config.balance_reward,
          ),
      ])

  elif agent == "msf":
# USFA + cumulants from FARM + Q-learning
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.QAuxConfig(),
        configs.ModularUSFAConfig(),
        configs.RewardConfig(),
        configs.FarmConfig(),
      ],
      dict_configs=default_config)

    return kitchen_helpers.msf(
      config,
      env_spec,
      NetworkCls=nets.msf,
      predict_cumulants=True,
      learn_model=False,
      use_separate_eval=True,
      task_embedding=config.task_embedding)

  else:
    raise NotImplementedError(agent)

  loss_label=None
  eval_network=False
  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
