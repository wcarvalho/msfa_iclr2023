import functools
from acme import wrappers
from utils import ObservationRemapWrapper
from agents import td_agent
from envs.acme.multiroom_goto import MultiroomGoto
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
import tensorflow as tf
import dm_env
from utils import data as data_utils
from agents.td_agent import losses
from losses import usfa as usfa_losses
from losses import cumulants


# -----------------------
# specific to these set of experiments
# -----------------------
from projects.colocation import nets
from projects.colocation import configs

def make_environment_sanity_check(evaluation: bool = False, simple: bool = True, agent='r2d1', nowalls: bool = False, one_room:bool=False, deterministic_rooms:bool=False, room_reward: float = 0.0):
    if simple:
        objs = [{'pan': 1}, {'tomato': 1}, {'knife':1}]
    else:
        objs = [{'pan': 1,'pot':1,'stove':1}, {'tomato': 1,'lettuce':1, 'onion':1}, {'knife':1,'apple':1, 'orange':1}]

    unique_objs = list(set(functools.reduce(lambda x,y: x + list(y.keys()),objs,[])))

    env = MultiroomGoto(
        objs,
        agent_view_size=5,
        mission_objects=unique_objs,
        pickup_required=True,
        tile_size=8,
        epsilon=0.0,
        room_size=5,
        doors_start_open=True,
        stop_when_gone=True,
        walls_gone=nowalls,
        one_room=one_room,
        deterministic_rooms=deterministic_rooms,
        room_reward = room_reward,
        wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=8)]
    )

    wrapper_list = [
        functools.partial(ObservationRemapWrapper,
                          remap=dict(mission='task', pickup='state_features')),
        wrappers.ObservationActionRewardWrapper,
        wrappers.SinglePrecisionWrapper,
    ]
    return wrappers.wrap_all(env, wrapper_list)

def q_aux_loss(config):
  """Create auxilliary Q-learning loss for SF
  """
  if config.q_aux == "single":
    loss = usfa_losses.QLearningAuxLoss
  elif config.q_aux == "ensemble":
    loss = usfa_losses.QLearningEnsembleAuxLoss
  else:
    raise RuntimeError(config.q_aux)

  return loss(
          coeff=config.value_coeff,
          discount=config.discount,
          sched_end=config.q_aux_anneal,
          sched_end_val=config.q_aux_end_val,
          tx_pair=config.tx_pair)

def load_agent_settings_sanity_check(env_spec, config_kwargs=None, agent = "r2d1", train_task_as_z = None):
    default_config = dict()
    default_config.update(config_kwargs or {})
    if agent=='r2d1':
        config = configs.R2D1Config(**default_config)

        NetworkCls=nets.r2d1 # default: 2M params
        NetKwargs=dict(config=config, env_spec=env_spec)
        LossFn = td_agent.R2D2Learning
        LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
        loss_label = 'r2d1'
        eval_network= config.eval_network
    elif agent=='r2d1_noise':
        config = data_utils.merge_configs(
            dataclass_configs=[configs.NoiseConfig()],
            dict_configs=default_config
        )

        NetworkCls = nets.r2d1_noise  # default: 2M params
        NetKwargs = dict(config=config, env_spec=env_spec)
        LossFn = td_agent.R2D2Learning
        LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
        loss_label = 'r2d1'
        eval_network = config.eval_network
    elif agent=='usfa':
        config = data_utils.merge_configs(
            dataclass_configs=[configs.USFAConfig()],
            dict_configs=default_config
        )

        NetworkCls = nets.usfa  # default: 2M params
        NetKwargs = dict(
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
    elif agent == 'usfa_conv':
        config = data_utils.merge_configs(
            dataclass_configs=[
                configs.USFAConfig(),
                configs.QAuxConfig(),
                configs.RewardConfig()],
            dict_configs=default_config
        )

        NetworkCls = nets.usfa  # default: 2M params
        NetKwargs = dict(
            config=config,
            env_spec=env_spec,
            use_seperate_eval=True,
            predict_cumulants=True,
            cuulant_type='conv',
            train_task_as_z=train_task_as_z)
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
                q_aux_loss(config),
                cumulants.CumulantRewardLoss(
                    shorten_data_for_cumulant=True,
                    coeff=config.reward_coeff,
                    loss=config.reward_loss,
                    balance=config.balance_reward,
                ),
            ])
        loss_label = 'usfa'
        eval_network = config.eval_network
    elif agent == 'usfa_lstm':
        config = data_utils.merge_configs(
            dataclass_configs=[
                configs.USFAConfig(),
                configs.QAuxConfig(),
                configs.RewardConfig()],
            dict_configs=default_config
        )

        NetworkCls = nets.usfa  # default: 2M params
        NetKwargs = dict(
            config=config,
            env_spec=env_spec,
            use_seperate_eval=True,
            predict_cumulants=True,
            cumulant_type='lstm',
            train_task_as_z=train_task_as_z)
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
                q_aux_loss(config),
                cumulants.CumulantRewardLoss(
                    shorten_data_for_cumulant=True,
                    coeff=config.reward_coeff,
                    loss=config.reward_loss,
                    balance=config.balance_reward,
                ),
            ])
        loss_label = 'usfa'
        eval_network = config.eval_network
    else:
        raise ValueError("Please specify a valid agent type")



    return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network



