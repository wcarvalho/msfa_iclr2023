from typing import NamedTuple

import acme
import functools

from acme import types
from acme import wrappers
import dm_env
import rlax

from envs.acme.goto_avoid import GoToAvoid
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper

from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses import usfa as usfa_losses
from losses.vae import VaeAuxLoss
from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss
from losses import cumulants
from modules.ensembles import QLearningEnsembleLoss

from projects.kitchen_combo import borsa_configs
from projects.common_usfm import agent_loading

class GotoObs(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  image: types.Nest
  pickup: types.Nest
  mission: types.Nest
  train_tasks: types.Nest

def make_environment(evaluation: bool = False,
                     tile_size=8,
                     setting='',
                     path='.',
                     image_wrapper=True,
                     obj2rew=None,
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
  setting = setting or 'xl_respawn'
  settings = dict(
    # -----------------------
    # small
    # -----------------------
    # small=dict(room_size=5, nobjects=1),
    # small_nopickup=dict(
    #   room_size=5, nobjects=1,
    #   pickup_required=False),
    # # -----------------------
    # # medium
    # # -----------------------
    # medium=dict(room_size=8, nobjects=2),
    # medium_nopickup=dict(
    #   room_size=8, nobjects=2,
    #   pickup_required=False),
    # # -----------------------
    # # large
    # # -----------------------
    # large=dict(room_size=8, nobjects=3),
    # large_nopickup=dict(
    #   room_size=9, nobjects=3,
    #   pickup_required=False),
    # large_respawn=dict(
    #   room_size=9, nobjects=3,
    #   respawn=True),
    xl_respawn=dict(
      room_size=10, nobjects=3,
      respawn=True),
    # xxl_nopickup=dict(
    #   room_size=11, nobjects=3,
    #   pickup_required=False),
    # xxl_nopickup_respawn=dict(
    #   room_size=11, nobjects=3,
    #   respawn=True,
    #   pickup_required=False),
    )
  if obj2rew is None:
    if evaluation:
      obj2rew={
          # "A.Train|1,0,0,0|":{
          #     "pan" : 1,
          #     "plates" : 0,
          #     "tomato" : 0,
          #     "knife" : 0,
          #     },
          # "A.Train|0,1,0,0|":{
          #     "pan" : 0,
          #     "plates" : 1,
          #     "tomato" : 0,
          #     "knife" : 0,
          #     },
          # "A.Train|0,0,1,0|":{
          #     "pan" : 0,
          #     "plates" : 0,
          #     "tomato" : 1,
          #     "knife" : 0,
          #     },
          # "A.Train|0,0,0,1|":{
          #     "pan" : 0,
          #     "plates" : 0,
          #     "tomato" : 0,
          #     "knife" : 1,
          #     },
          'B.Test|1,1,0,0|':{
              "pan" : 1,
              "plates" :1,
              "tomato" : 0,
              "knife" : 0,
              },
          'B.Test|1,1,.5,.5|':{
              "pan" : 1,
              "plates" : 1,
              "tomato" : .5,
              "knife" : .5,
              },
          'B.Test|1,1,1,1|':{
              "pan" : 1,
              "plates" : 1,
              "tomato" : 1,
              "knife" : 1,
              },
          'B.Test|-1,1,0,1|':{
              "pan" : -1,
              "plates" : 1,
              "tomato" : 0,
              "knife" : 1,
              },
          'B.Test|-1,1,-1,.5|':{
              "pan" : -1,
              "plates" : 1,
              "tomato" : -1,
              "knife" : .5,
              },
          'B.Test|-1,1,-1,1|':{
              "pan" : -1,
              "plates" : 1,
              "tomato" : -1,
              "knife" : 1,
              },
          'B.Test|-1,1,-1,-1|':{
              "pan" : -1,
              "plates" : 1,
              "tomato" : -1,
              "knife" : -1,
              },
          'B.Test|-.5,1,-.5,-.5|':{
              "pan" : -.5,
              "plates" : 1,
              "tomato" : -.5,
              "knife" : -.5,
              },
          # 'B.Test|-1,-1,-1,-1|':{
          #     "pan" : -1,
          #     "plates" : -1,
          #     "tomato" : -1,
          #     "knife" : -1,
          #     },
      }
    else:
      obj2rew={
          "1,0,0,0":{
              "pan" : 1,
              "plates" : 0,
              "tomato" : 0,
              "knife" : 0,
              },
          "0,1,0,0":{
              "pan" : 0,
              "plates" : 1,
              "tomato" : 0,
              "knife" : 0,
              },
          "0,0,1,0":{
              "pan" : 0,
              "plates" : 0,
              "tomato" : 1,
              "knife" : 0,
              },
          "0,0,0,1":{
              "pan" : 0,
              "plates" : 0,
              "tomato" : 0,
              "knife" : 1,
              },
      }

  env_wrappers = []
  if image_wrapper:
    env_wrappers.append(functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size))

  env = GoToAvoid(
    tile_size=tile_size,
    obj2rew=obj2rew,
    path=path,
    **settings[setting],
    wrappers=env_wrappers,
    ObsCls=GotoObs,
    train_tasks_obs=True,
    )

  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(
            pickup='state_features',
            mission='task',
            )),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)


def load_agent_settings(agent, env_spec, config_kwargs=None, env_kwargs=None):

  return agent_loading.default_agent_settings(agent=agent,
    env_spec=env_spec,
    configs=borsa_configs,
    config_kwargs=config_kwargs,
    env_kwargs=env_kwargs)
