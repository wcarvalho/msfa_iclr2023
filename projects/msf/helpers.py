import acme
import functools

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

from projects.msf import nets
from projects.msf import configs


def make_environment(evaluation: bool = False,
                     tile_size=8,
                     setting='small',
                     path='.',
                     image_wrapper=True,
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
    small_nopickup=dict(
      room_size=5, nobjects=1,
      pickup_required=False),
    medium=dict(room_size=8, nobjects=2),
    medium_nopickup=dict(
      room_size=8, nobjects=2,
      pickup_required=False),
    large=dict(room_size=8, nobjects=3),
    large_nopickup=dict(
      room_size=8, nobjects=3,
      pickup_required=False),
    large_respawn=dict(
      room_size=9, nobjects=3,
      respawn=True),
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

def q_aux_sf_loss(config):
  """Create auxilliary Q-learning loss for SF
  """
  if config.q_aux == "single":
    loss = usfa_losses.QLearningAuxLoss
  elif config.q_aux == "ensemble":
    loss = usfa_losses.QLearningEnsembleAuxLoss
  else:
    raise RuntimeError(config.q_aux)

  if config.sf_loss == 'n_step_q_learning':
    tx_pair = rlax.IDENTITY_PAIR
  elif config.sf_loss == 'transformed_n_step_q_learning':
    tx_pair = rlax.SIGNED_HYPERBOLIC_PAIR
  else:
    raise NotImplementedError(config.sf_loss)

  return loss(
          coeff=config.value_coeff,
          discount=config.discount,
          sched_end=config.q_aux_anneal,
          sched_end_val=config.q_aux_end_val,
          tx_pair=tx_pair)

def usfa_farm(default_config, env_spec, net='flat', predict_cumulants=True, learn_model=False):
  config = data_utils.merge_configs(
    dataclass_configs=[
      configs.ModularUSFAConfig(),
      configs.QAuxConfig(),
      configs.RewardConfig(),
      configs.FarmModelConfig() if learn_model else configs.FarmConfig(),
    ],
    dict_configs=default_config)

  if net == "flat":
    NetworkCls =  nets.usfa_farmflat_model
  elif net == "independent":
    NetworkCls =  nets.usfa_farm_model
  elif net == "msf":
    NetworkCls =  nets.msf
  else:
    raise NotImplementedError

  NetKwargs=dict(
    config=config,
    env_spec=env_spec,
    predict_cumulants=predict_cumulants,
    learn_model=learn_model)

  LossFn = td_agent.USFALearning

  aux_tasks=[q_aux_sf_loss(config)]

  if predict_cumulants:
    aux_tasks.append(
      cumulants.CumulantRewardLoss(
        shorten_data_for_cumulant=True,
        coeff=config.reward_coeff,
        loss=config.reward_loss,
        balance=config.balance_reward))

  if learn_model:
    if config.contrast_module_coeff > 0:
      aux_tasks.append(
          ModuleContrastLoss(
            coeff=config.contrast_module_coeff,
            extra_negatives=config.extra_module_negatives,
            temperature=config.temperature,
            prediction=config.contrast_module_pred)
          )
    if config.contrast_time_coeff > 0:
      aux_tasks.append(
          TimeContrastLoss(
            coeff=config.contrast_time_coeff,
            extra_negatives=config.extra_time_negatives,
            temperature=config.temperature,
            normalize_step=config.normalize_step)
          )
  LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
  LossFnKwargs.update(
    loss=config.sf_loss,
    shorten_data_for_cumulant=True, # needed since using delta for cumulant
    extract_cumulants=losses.cumulants_from_preds,
    aux_tasks=aux_tasks)

  loss_label = 'usfa'
  eval_network = config.eval_network

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network


def load_agent_settings(agent, env_spec, config_kwargs=None, setting='small'):
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
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_no_task": 
  # UVFA + noise added to goal embedding
    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config()],
      dict_configs=default_config
    )

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec, task_input=False)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_noise_eval": 
  # UVFA + noise added to goal embedding
    config = data_utils.merge_configs(
      dataclass_configs=[configs.NoiseConfig()],
      dict_configs=default_config
    )

    NetworkCls=nets.r2d1_noise # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec, eval_noise=True)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_farm":
  # UVFA + FARM

    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config(), configs.FarmConfig()],
      dict_configs=default_config
      )
    NetworkCls=nets.r2d1_farm # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)

    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "usfa":
  # USFA

    config = data_utils.merge_configs(
      dataclass_configs=[configs.USFAConfig()],
      dict_configs=default_config
      )

    NetworkCls=nets.usfa # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      use_seperate_eval=True)

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.sf_loss,
      lambda_=config.lambda_,
      )

    loss_label = 'usfa'
    eval_network = config.eval_network

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
      use_seperate_eval=True,
      predict_cumulants=True)

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.sf_loss,
      shorten_data_for_cumulant=True,
      extract_cumulants=functools.partial(
        losses.cumulants_from_preds,
        stop_grad=True,
      ),
      aux_tasks=[
        q_aux_sf_loss(config),
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,
          loss=config.reward_loss,
          balance=config.balance_reward,
          ),
      ])

    loss_label = 'usfa'
    eval_network = config.eval_network

  elif agent == "msf":
  # USFA + cumulants from FARM + Q-learning
    return usfa_farm(default_config, env_spec,
      net='msf',
      predict_cumulants=True,
      learn_model=True)
  elif agent == "msf_delta_model":
  # USFA + cumulants from FARM + Q-learning
    default_config['contrast_module_pred'] = 'delta'
    return usfa_farm(default_config, env_spec,
      net='msf',
      predict_cumulants=True,
      learn_model=True)
  elif agent == "msf_time_model":
  # USFA + cumulants from FARM + Q-learning
    return usfa_farm(default_config, env_spec,
      net='msf',
      predict_cumulants=True,
      learn_model=True)
  elif agent == "msf_state_model":
  # USFA + cumulants from FARM + Q-learning
    default_config['contrast_module_pred'] = 'state'
    return usfa_farm(default_config, env_spec,
      net='msf',
      predict_cumulants=True,
      learn_model=True)
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
