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
from projects.common_usfm import agent_loading
from projects.common_usfm import nets as common_nets

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
  setting = setting or 'small'
  settings = dict(
    # -----------------------
    # small
    # -----------------------
    small=dict(room_size=5, nobjects=1),
    small_nopickup=dict(
      room_size=5, nobjects=1,
      pickup_required=False),
    # -----------------------
    # medium
    # -----------------------
    medium=dict(room_size=8, nobjects=2),
    medium_nopickup=dict(
      room_size=8, nobjects=2,
      pickup_required=False),
    # -----------------------
    # large
    # -----------------------
    large=dict(room_size=8, nobjects=3),
    large_nopickup=dict(
      room_size=9, nobjects=3,
      pickup_required=False),
    large_respawn=dict(
      room_size=9, nobjects=3,
      respawn=True),
    xl_respawn=dict(
      room_size=10, nobjects=3,
      respawn=True),
    xxl_nopickup=dict(
      room_size=11, nobjects=3,
      pickup_required=False),
    xxl_nopickup_respawn=dict(
      room_size=11, nobjects=3,
      respawn=True,
      pickup_required=False),
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

  add_bias = getattr(config, "step_penalty", 0) > 0
  return loss(
          coeff=config.value_coeff,
          discount=config.discount,
          sched_end=config.q_aux_anneal,
          sched_end_val=config.q_aux_end_val,
          tx_pair=tx_pair,
          add_bias=add_bias,
          mask_loss=config.qaux_mask_loss,
          target_w=config.target_phi,
          stop_w_grad=getattr(config, 'stop_w_grad', False))


def load_agent_settings(agent, env_spec, config_kwargs=None, setting='small'):
  default_config = dict()
  default_config.update(config_kwargs or {})
  agent = agent.lower()

  loss_label=''
  eval_network=True

  if agent == "r2d1":
  # Recurrent DQN/UVFA
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[
        configs.R2D1Config(),
      ],
      task_input='qfn_concat',
      task_embedding='none',
      )

  elif agent == "r2d1_no_task": 
  # UVFA + noise added to goal embedding
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[
        configs.R2D1Config(),
      ],
      task_input='none',
      task_embedding='none',
      )


  elif agent == "r2d1_noise_eval": 
  # UVFA + noise added to goal embedding
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      NetworkCls=common_nets.r2d1_noise,
      dataclass_configs=[
        configs.R2D1Config(),
      ],
      )

  elif agent == "r2d1_farm":
  # UVFA + FARM
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[
        configs.R2D1Config(),
        configs.FarmConfig(),
      ],
    )
    NetworkCls=nets.r2d1_farm # default: 2M params

  elif agent == "usfa":
  # USFA
    config, _, NetKwargs, LossFn, LossFnKwargs = agent_loading.usfa_lstm(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          configs.USFAConfig(),
          ],
        predict_cumulants=False,
      )
    NetworkCls=nets.usfa # default: 2M params

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning

    config, _, NetKwargs, LossFn, LossFnKwargs = agent_loading.usfa_lstm(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          configs.QAuxConfig(),
          configs.RewardConfig(),
          configs.USFAConfig(),
          ],
      )

    NetworkCls=nets.usfa # default: 2M params

  elif agent == "msf":
  # USFA + cumulants from FARM + Q-learning
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.msf(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[
        configs.QAuxConfig(),
        configs.RewardConfig(),
        configs.FarmConfig(),
        configs.ModularUSFAConfig(),
      ],
    )

    NetworkCls=nets.msf # default: 2M params

  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
