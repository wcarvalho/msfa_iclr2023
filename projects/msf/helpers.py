import acme
import functools

from acme import wrappers
import dm_env

from envs.acme.goto_avoid import GoToAvoid
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper

from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses.usfa import ValueAuxLoss
from losses.vae import VaeAuxLoss
from losses.contrastive_model import DeltaContrastLoss
from losses import cumulants

from projects.msf import nets
from projects.msf import configs


def make_environment(evaluation: bool = False,
                     tile_size=8,
                     path='.',
                     ) -> dm_env.Environment:
  """Loads environments."""
  if evaluation:
    obj2rew={
        'pan_plates':{
            "pan" : 1,
            "plates" :1,
            "fork" : 0,
            "knife" : 0,
            },
        'all':{
            "pan" : 1,
            "plates" : 1,
            "fork" : 1,
            "knife" : 1,
            },
        'mix1':{
            "pan" : -1,
            "plates" : 1,
            "fork" : -1,
            "knife" : 1,
            },
        'mix2':{
            "pan" : -1,
            "plates" : 1,
            "fork" : 0,
            "knife" : 1,
            },
    }
  else:
    obj2rew=dict(
        pan={
            "pan" : 1,
            "plates" : 0,
            "fork" : 0,
            "knife" : 0,
            },
        plates={
            "pan" : 0,
            "plates" : 1,
            "fork" : 0,
            "knife" : 0,
            },
        fork={
            "pan" : 0,
            "plates" : 0,
            "fork" : 1,
            "knife" : 0,
            },
        knife={
            "pan" : 0,
            "plates" : 0,
            "fork" : 0,
            "knife" : 1,
            },
    )

  env = GoToAvoid(
    tile_size=tile_size,
    obj2rew=obj2rew,
    path=path,
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


def load_agent_settings(agent, env_spec, config_kwargs=None):
  default_config = dict()
  default_config.update(config_kwargs or {})

  if agent == "r2d1": # Recurrent DQN
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'r2d1'

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

  elif agent == "r2d1_vae":
    # R2D1 + VAE
    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config(), configs.VAEConfig()],
      dict_configs=default_config
      )

    NetworkCls =  nets.r2d1_vae # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[VaeAuxLoss(
                  coeff=config.vae_coeff,
                  beta=config.beta)])

    loss_label = 'r2d1'

  elif agent == "r2d1_farm_model":

    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.R2D1Config(), configs.FarmConfig(), configs.FarmModelConfig()],
      dict_configs=default_config
      )

    NetworkCls=nets.r2d1_farm_model # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[DeltaContrastLoss(
                    coeff=config.model_coeff,
                    extra_negatives=config.extra_negatives,
                    temperature=config.temperature,
                  )])

    loss_label = 'r2d1'

  elif agent == "usfa": # Universal Successor Features
    state_dim = env_spec.observations.observation.state_features.shape[0]

    config = configs.USFAConfig(**default_config)
    config.state_dim = state_dim

    NetworkCls=nets.usfa # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)


    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'usfa'

  elif agent == "usfa_reward_vae":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFARewardVAEConfig(**default_config)

    NetworkCls =  nets.usfa_reward_vae # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulants=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=[
        VaeAuxLoss(coeff=config.vae_coeff),
        cumulants.CumulantRewardLoss(
          coeff=config.reward_coeff,  # coefficient for loss
          loss=config.reward_loss),  # type of loss for reward
      ])   # type of loss for reward

    loss_label = 'usfa'

  elif agent == "usfa_farmflat_model":
    # Universal Successor Features which learns cumulants with structured transition model
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.USFAConfig(), configs.FarmConfig(), configs.FarmModelConfig(), configs.RewardConfig()],
      dict_configs=default_config
      )

    NetworkCls =  nets.usfa_farmflat_model
    NetKwargs=dict(config=config,env_spec=env_spec)
    
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulants=losses.cumulants_from_preds,
      shorten_data_for_cumulant=True, # needed since using delta for cumulant
      aux_tasks=[
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,  # coefficient for loss
          loss=config.reward_loss),  # type of loss for reward
        DeltaContrastLoss(
                    coeff=config.model_coeff,
                    extra_negatives=config.extra_negatives,
                    temperature=config.temperature),
      ])
    loss_label = 'usfa'

  elif agent == "usfa_farm_model":
    # Universal Successor Features which learns cumulants with structured transition model
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.ModularUSFAConfig(), configs.FarmConfig(), configs.FarmModelConfig(), configs.RewardConfig()],
      dict_configs=default_config
      )

    NetworkCls =  nets.usfa_farm_model
    NetKwargs=dict(config=config,env_spec=env_spec)
    
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulants=losses.cumulants_from_preds,
      shorten_data_for_cumulant=True, # needed since using delta for cumulant
      aux_tasks=[
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,  # coefficient for loss
          loss=config.reward_loss),  # type of loss for reward
        DeltaContrastLoss(
                    coeff=config.model_coeff,
                    extra_negatives=config.extra_negatives,
                    temperature=config.temperature),
      ])
    loss_label = 'usfa'
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label
