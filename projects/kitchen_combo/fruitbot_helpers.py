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


from projects.msf import nets
from projects.kitchen_gridworld import helpers as kitchen_helpers
from projects.msf import helpers as msf_helpers
from projects.kitchen_combo import fruitbot_configs

from envs.procgen_gym_task import ProcgenGymTask, ProcGenMultitask

def make_environment(
  setting='',
  evaluation: bool = False,
  **kwargs) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  setting = setting or 'taskgen_long_easy'
  assert setting in [
    'procgen_easy',
    'procgen_easy_medium',
    'procgen_easy_hard',
    'procgen_hard',
    'taskgen_short_easy',
    'taskgen_short_hard',
    'taskgen_long_easy',
    'taskgen_long_hard']

  # -----------------------
  # environments
  # -----------------------
  if 'taskgen_short' in setting:
    if evaluation:
      all_level_kwargs={
        'b.eval|-1,-1|': dict(
          env='fruitbotnn', task=[-1,-1]),
        'b.eval|-1,1|': dict(
          env='fruitbotnp', task=[-1,1]),
        'b.eval|1,-1|': dict(
          env='fruitbotpn', task=[1,-1]),
        'b.eval|1,1|': dict(
          env='fruitbotpp', task=[1,1]),
        'a.train|0,1|': dict(
          env='fruitbotzp', task=[0,1]),
        'a.train|1,0|': dict(
          env='fruitbotpz', task=[1,0])
      }
    else:
      all_level_kwargs={
        '|0,1|': dict(
          env='fruitbotzp', task=[0,1]),
        '|1,0|': dict(
          env='fruitbotpz', task=[1,0])
      }
    if 'easy' in setting:
      num_levels=200
    elif 'hard' in setting:
      num_levels=500
  elif 'taskgen_long' in setting:
    train_level_kwargs={
        'a.train|1,0,0,0|': dict(
          env='wilkabotpzzz', task=[1,0,0,0]),
        'a.train|0,1,0,0|': dict(
          env='wilkabotzpzz', task=[0,1,0,0]),
        'a.train|0,0,1,0|': dict(
          env='wilkabotzzpz', task=[0,0,1,0]),
        'a.train|0,0,0,1|': dict(
          env='wilkabotzzzp', task=[0,0,0,1]),
      }
    if evaluation:
      all_level_kwargs={
        'b.eval|1,1,1,1|': dict(
          env='wilkabotpppp', task=[1,1,1,1]),
        'b.eval|1,-1,-1,-1|': dict(
          env='wilkabotpnnn', task=[1,-1,-1,-1]),
        'b.eval|1,0,-1,-1|': dict(
          env='wilkabotpznn', task=[1,0,-1,-1]),
        'b.eval|1,0,0,-1|': dict(
          env='wilkabotpzzn', task=[1,0,0,-1]),
        **train_level_kwargs,
      }
    else:
      all_level_kwargs=train_level_kwargs
    if 'easy' in setting:
      num_levels=200
    elif 'hard' in setting:
      num_levels=500
  elif 'procgen' in setting:
    all_level_kwargs={
        '1,-1': dict(
          env='fruitbot', task=[1,1]), # ignore it
      }
    if setting == 'procgen_easy':
      setting = 'easy'
      num_levels=200
    elif setting == 'procgen_easy_medium':
      setting = 'easy'
      num_levels=100
    elif setting == 'procgen_easy_hard':
      setting = 'easy'
      num_levels=50
    elif setting == 'procgen_hard':
      setting = 'hard'
      num_levels=500
    else:
      raise NotImplementedError(setting)
  # -----------------------
  # num levels
  # -----------------------
  if evaluation:
    num_levels=0

  # -----------------------
  # settting
  # -----------------------
  setting = 'easy' if 'easy' in setting else 'hard'

  env = ProcGenMultitask(
    all_level_kwargs=all_level_kwargs,
    EnvCls=ProcgenGymTask,
    distribution_mode=setting,
    num_levels=num_levels,
    )

  wrapper_list = [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)



def load_agent_settings(agent, env_spec, config_kwargs=None, env_kwargs=None):
  default_config = dict()
  default_config.update(config_kwargs or {})

  agent = agent.lower()
  if agent == "r2d1":
  # Recurrent DQN/UVFA
    config = data_utils.merge_configs(
      dataclass_configs=[fruitbot_configs.R2D1Config()],
      dict_configs=default_config
    )

    NetworkCls=nets.r2d1 # default: (1.96M) params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning

    config = data_utils.merge_configs(
      dataclass_configs=[
        fruitbot_configs.USFAConfig(),
        fruitbot_configs.QAuxConfig(),
        fruitbot_configs.RewardConfig()],
      dict_configs=default_config
      )

    NetworkCls=nets.usfa # default: (1.96M) params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      use_seperate_eval=True,
      predict_cumulants=True)

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
  # USFA + cumulants from FARM + Q-learning (1.9M)
    config = data_utils.merge_configs(
      dataclass_configs=[
        fruitbot_configs.QAuxConfig(),
        fruitbot_configs.ModularUSFAConfig(),
        fruitbot_configs.RewardConfig(),
        fruitbot_configs.FarmConfig(),
      ],
      dict_configs=default_config)

    return kitchen_helpers.msf(
      config,
      env_spec,
      NetKwargs=dict(
        config=config,
        env_spec=env_spec,
        predict_cumulants=True,
        learn_model=False),
      NetworkCls=nets.msf,
      predict_cumulants=True,
      learn_model=False)

  else:
    raise NotImplementedError(agent)

  loss_label=None
  eval_network=False
  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
