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
from projects.kitchen_combo import combo_configs
from projects.common_usfm import agent_loading

class ComboObsTuple(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  mission: types.Nest
  train_tasks: types.Nest

def make_environment(evaluation: bool = False,
                     tile_size=8,
                     setting='',
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
  setting = setting or 'test_remove'
  settings = dict(
    test_remove=dict(room_size=5, ntasks=1, task_reset_behavior='remove_all'),
    test_respawn=dict(room_size=5, ntasks=1, task_reset_behavior='respawn'),
    small_remove=dict(room_size=7, ntasks=1, task_reset_behavior='remove_all'),
    small_respawn=dict(room_size=7, ntasks=1, task_reset_behavior='respawn'),
    medium_remove=dict(room_size=9, ntasks=1, task_reset_behavior='remove_all'),
    medium_respawn=dict(room_size=9, ntasks=1, task_reset_behavior='respawn')
  )
  assert setting in settings.keys()
  task2arguments=dict(
      toggle=dict(x=['microwave', 'stove']),
      pickup=dict(x=['knife', 'fork']),
      slice_putdown=dict(x=['potato', 'apple', 'orange']),
      clean=dict(x=['pot', 'pan', 'plates']), # also uses stove
      chill=dict(x=['lettuce', 'onion', 'tomato']),
  )
  if task2rew is None:
    train = {
          "a.Train.1.toggle":{
              "slice_putdown" : 1,
              "toggle" : 0,
              "clean" : 0,
              "chill" : 0,
              },
          "a.Train.2.slice":{
              "slice_putdown" : 0,
              "toggle" : 1,
              "clean" : 0,
              "chill" : 0,
              },
          "a.Train.3.clean":{
              "slice_putdown" : 0,
              "toggle" : 0,
              "clean" : 1,
              "chill" : 0,
              },
          "a.Train.4.chill":{
              "slice_putdown" : 0,
              "toggle" : 0,
              "clean" : 0,
              "chill" : 1,
              }
      }
    if evaluation:
      task2rew={
          **train,
          "b.Eval.2.slice-toggle":{
              "slice_putdown" : 1,
              "toggle" : 1,
              "clean" : 0,
              "chill" : 0,
              },
          "b.Eval.2.slice-clean":{
              "slice_putdown" : 1,
              "toggle" : 0,
              "clean" : 1,
              "chill" : 0,
              },
          "b.Eval.2.toggle-clean":{
              "slice_putdown" : 0,
              "toggle" : 1,
              "clean" : 1,
              "chill" : 0,
              },
          "b.Eval.2.chill-clean":{
              "slice_putdown" : 0,
              "toggle" : 0,
              "clean" : 1,
              "chill" : 1,
              },
          "b.Eval.3.slice-toggle-clean":{
              "slice_putdown" : 1,
              "toggle" : 1,
              "clean" : 1,
              "chill" : 0,
              },
          "b.Eval.3.toggle-clean-chill":{
              "slice_putdown" : 0,
              "toggle" : 1,
              "clean" : 1,
              "chill" : 1,
              },
          "b.Eval.3.slice-toggle-clean":{
              "slice_putdown" : 1,
              "toggle" : 0,
              "clean" : 1,
              "chill" : 1,
              },
          "b.Eval.4.slice-toggle-clean-chill":{
              "slice_putdown" : 1,
              "toggle" : 1,
              "clean" : 1,
              "chill" : 1,
              },
      }
    else:
      task2rew=train

  if 'test' in setting:
    train={
      '0.pickup':{"pickup" : 1, 'toggle': 0},
      '0.toggle':{"pickup" : 0, 'toggle': 1},
    }
    if evaluation:
      task2rew={
        **train,
        '1.|+|+|':{"pickup" : 1, 'toggle': 1},
        '2.|+|-|':{"pickup" : 1, 'toggle': -1},
        '2.|-|+|':{"pickup" : -1, 'toggle': 1},
        '3.|-|0|':{"pickup" : -1, 'toggle': 0},
        '3.|0|-|':{"pickup" : 0, 'toggle': -1},
      }
    else:
      task2rew=train


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
    task2arguments=task2arguments,
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
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[
        combo_configs.R2D1Config(),
      ])

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.usfa_lstm(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          combo_configs.QAuxConfig(),
          combo_configs.RewardConfig(),
          combo_configs.USFAConfig(),
          ],
      )

  elif agent == "msf":
# USFA + cumulants from FARM + Q-learning
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.msf(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          combo_configs.QAuxConfig(),
          combo_configs.RewardConfig(),
          combo_configs.ModularUSFAConfig(),
          combo_configs.FarmConfig(),
        ],
      )
  else:
    raise NotImplementedError(agent)

  loss_label=None
  eval_network=False
  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
