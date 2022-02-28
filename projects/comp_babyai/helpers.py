import acme
import functools

from acme import wrappers
import dm_env


from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses import usfa as usfa_losses
from losses.vae import VaeAuxLoss
from losses.contrastive_model import DeltaContrastLoss
from losses import cumulants
from modules.ensembles import QLearningEnsembleLoss

from projects.comp_babyai import nets
from projects.comp_babyai import configs

from envs.acme.goto_avoid import GoToAvoid
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor


def make_environment(evaluation: bool = False,
                     tile_size=8,
                     path='.',
                     task_kinds=None,
                     room_size=7,
                     ) -> dm_env.Environment:
  """Loads environments."""
  task_kinds = task_kinds or [
    'pickup',
    'place',
    # 'heat',
    'pickup_cleaned',
    'pickup_sliced',
    'pickup_chilled',
    'pickup_cooked',
  ]

  instr_preproc = InstructionsPreprocessor(
    path="data/babyai_kitchen/vocab.json")

  env = MultitaskKitchen(
    task_kinds=task_kinds,
    tile_size=tile_size,
    path=path,
    room_size=room_size,
    wrappers=[
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size),
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc,
        max_length=30)],
    )

  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(
            # pickup='state_features',
            mission='task',
            )),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)

def make_environment(evaluation: bool = False,
                     tile_size=8,
                     setting='small',
                     path='.',
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
    small=dict(room_size=5, nobjects=1),
    medium=dict(room_size=8, nobjects=2),
    large=dict(room_size=10, nobjects=3),
    )
  if evaluation:
    obj2rew={
        '1,1,0,0':{
            "pan" : 1,
            "plates" :1,
            "tomato" : 0,
            "knife" : 0,
            },
        '1,1,1,1':{
            "pan" : 1,
            "plates" : 1,
            "tomato" : 1,
            "knife" : 1,
            },
        '-1,1,-1,1':{
            "pan" : -1,
            "plates" : 1,
            "tomato" : -1,
            "knife" : 1,
            },
        '-1,1,0,1':{
            "pan" : -1,
            "plates" : 1,
            "tomato" : 0,
            "knife" : 1,
            },
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

  env = GoToAvoid(
    tile_size=tile_size,
    obj2rew=obj2rew,
    path=path,
    **settings[setting],
    wrappers=[functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size)]
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


def load_agent_settings(agent, env_spec, config_kwargs=None, setting='small'):
  default_config = dict()
  default_config.update(config_kwargs or {})

  if agent == "r2d1": # Recurrent DQN
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_farm":

    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config(), configs.FarmConfig()],
      dict_configs=default_config
      )
    NetworkCls=nets.r2d1_farm # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'r2d1'
    eval_network = config.eval_network

  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
