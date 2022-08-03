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
from projects.common_usfm import agent_loading

from envs.procgen_gym_task import ProcgenGymTask, ProcGenMultitask

def make_environment(
  setting='',
  evaluation: bool = False,
  train_in_eval=False,
  max_episodes=3,
  completion_bonus=0.0,
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
    'taskgen_long_easy2',
    'taskgen_long_hard']

  # -----------------------
  # environments
  # -----------------------
  if 'taskgen_short' in setting:
    reward_coeff=1.0
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
        'b.eval|1,0,1,1|': dict(
          env='wilkabotpzpp', task=[1,0,1,1]),
        'b.eval|1,0,0,1|': dict(
          env='wilkabotpzzp', task=[1,0,0,1]),
        'b.eval|1,1,1,1|': dict(
          env='wilkabotpppp', task=[1,1,1,1]),
        'b.eval|1,-1,-1,-1|': dict(
          env='wilkabotpnnn', task=[1,-1,-1,-1]),
        'b.eval|1,0,-1,-1|': dict(
          env='wilkabotpznn', task=[1,0,-1,-1]),
        'b.eval|1,0,0,-1|': dict(
          env='wilkabotpzzn', task=[1,0,0,-1]),
      }
      if train_in_eval:
        all_level_kwargs.update(train_level_kwargs)
    else:
      all_level_kwargs=train_level_kwargs
    if setting == 'taskgen_long_easy':
      setting='easy'
      num_levels=200
    elif setting == 'taskgen_long_easy2':
      setting='easy'
      num_levels=500
    elif setting == 'taskgen_long_hard':
      setting='hard'
      num_levels=500
    else:
      raise NotImplementedError(setting)


  elif 'procgen' in setting:
    all_level_kwargs={
        '1,-1': dict(
          env='fruitbot', task=[1, 1]), # ignore it
      }
    max_episodes = 1
    completion_bonus = 0.0
    reward_coeff=0.1 # max reward = 10.0
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

  env = ProcGenMultitask(
    all_level_kwargs=all_level_kwargs,
    EnvCls=ProcgenGymTask,
    distribution_mode=setting,
    num_levels=num_levels,
    max_episodes=max_episodes,
    completion_bonus=completion_bonus,
    reward_coeff=reward_coeff,
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
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[fruitbot_configs.R2D1Config()],
      )

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning

    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.usfa_lstm(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          fruitbot_configs.QAuxConfig(),
          fruitbot_configs.RewardConfig(),
          fruitbot_configs.USFAConfig(),
          ],
      )


  elif agent == "msf":
  # USFA + cumulants from FARM + Q-learning (1.9M)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.msf(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          fruitbot_configs.QAuxConfig(),
          fruitbot_configs.RewardConfig(),
          fruitbot_configs.ModularUSFAConfig(),
          fruitbot_configs.FarmConfig(),
        ],
      )

  else:
    raise NotImplementedError(agent)

  loss_label=None
  eval_network=False
  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
