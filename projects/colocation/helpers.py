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
import rlax
from losses import msfa_stats
from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss


# -----------------------
# specific to these set of experiments
# -----------------------
from projects.colocation import nets
from projects.colocation import configs


"""
This is our main environment building function. Most of its arguments are just passed on to the MultiroomGoto environment
The only special arguments are:
     - simple (bool, Optional) If this is true, we only put one object in each room, if it is False, we have 3 objects per room
"""
def make_environment_sanity_check(
                                  simple: bool = False,
                                  nowalls: bool = False,
                                  one_room:bool=False,
                                  deterministic_rooms:bool=False,
                                  room_reward: float = 0.0,
                                  room_reward_task_vector: bool = True):

    #define objects we are working with
    if simple:
        objs = [{'pan': 1}, {'tomato': 1}, {'knife':1}]
    else:
        objs = [{'pan': 1,'pot':1,'bowl':1}, {'tomato': 1,'lettuce':1, 'onion':1}, {'knife':1,'apple':1, 'orange':1}]

    unique_objs = list(set(functools.reduce(lambda x,y: x + list(y.keys()),objs,[])))

    #build the environment
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
        room_reward_task_vector=room_reward_task_vector,
        wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=8)]
    )

    #super standard wrappers...
    wrapper_list = [
        functools.partial(ObservationRemapWrapper,
                          remap=dict(mission='task', pickup='state_features')),
        wrappers.ObservationActionRewardWrapper,
        wrappers.SinglePrecisionWrapper,
    ]
    return wrappers.wrap_all(env, wrapper_list)

#####################################
# Copied from kitchen_gridworld
#####################################
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

#####################################
# Copied from kitchen_gridworld
#####################################
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
          mask_loss=config.qaux_mask_loss)

#####################################
# Copied from kitchen_gridworld, but with
#####################################
def msf(config, env_spec, NetworkCls, use_separate_eval=True, predict_cumulants=True, learn_model=False, task_embedding='none'):

  NetKwargs=dict(
    config=config,
    env_spec=env_spec,
    predict_cumulants=predict_cumulants,
    learn_model=learn_model,
    task_embedding=task_embedding,
    use_separate_eval=use_separate_eval)

  LossFn = td_agent.USFALearning

  aux_tasks=[
    q_aux_sf_loss(config),
    msfa_stats.MsfaStats()
  ]

  if predict_cumulants:
    nmodules = config.nmodules if config.module_l1 else 1
    aux_tasks.append(
      cumulants.CumulantRewardLoss(
        shorten_data_for_cumulant=True,
        coeff=config.reward_coeff,
        loss=config.reward_loss,
        l1_coeff=config.phi_l1_coeff,
        wl1_coeff=config.w_l1_coeff,
        balance=config.balance_reward,
        reward_bias=config.step_penalty,
        nmodules=nmodules))

  cov_coeff = getattr(config, 'cov_coeff', None)

  if cov_coeff is not None:
    aux_tasks.append(
      cumulants.CumulantCovLoss(
        coeff=cov_coeff,
        blocks=config.nmodules,
        loss=config.cov_loss))

  if learn_model:
    if config.contrast_module_coeff > 0:
      aux_tasks.append(
          ModuleContrastLoss(
            coeff=config.contrast_module_coeff,
            extra_negatives=config.extra_module_negatives,
            temperature=config.temperature)
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
    mask_loss=config.sf_mask_loss,
    shorten_data_for_cumulant=True, # needed since using delta for cumulant
    extract_cumulants=losses.cumulants_from_preds,
    aux_tasks=aux_tasks)

  loss_label = 'usfa'
  eval_network = config.eval_network

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network

# Function to get agent settings for all relevant agents
# train_task_as_z determines if we use gaussian sampling or sample all train tasks in training
#   If it is True, we sample all train tasks
#   If it is False, we sample gaussian noise around w
#   If it is None, we set it equal to predict_cumulants

#task_embedding can be 'none' or 'vector', but only set it to 'vector' if you are also predicting cumulants
#recommended to always use vector when using MSF
def load_agent_settings_sanity_check(env_spec, config_kwargs=None, agent = "r2d1", train_task_as_z = None, task_embedding = 'none'):
    default_config = dict()
    default_config.update(config_kwargs or {})

    #R2D1
    if agent=='r2d1':
        config = configs.R2D1Config(**default_config)

        NetworkCls=nets.r2d1 # default: 2M params
        NetKwargs=dict(config=config, env_spec=env_spec)
        LossFn = td_agent.R2D2Learning
        LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
        loss_label = 'r2d1'
        eval_network= config.eval_network

    #R2D1 Noise
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

    #USFA with oracle cumulants
    elif agent=='usfa':
        config = data_utils.merge_configs(
            dataclass_configs=[configs.USFAConfig()],
            dict_configs=default_config
        )

        NetworkCls = nets.usfa  # default: 2M params
        NetKwargs = dict(
            config=config,
            env_spec=env_spec,
            use_seperate_eval=True,
            task_embed_type=task_embedding)

        LossFn = td_agent.USFALearning
        LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
        LossFnKwargs.update(
            loss=config.sf_loss,
            lambda_=config.lambda_,
        )

        loss_label = 'usfa'
        eval_network = config.eval_network

    #USFA with conv cumulants
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
            train_task_as_z=train_task_as_z,
            task_embed_type=task_embedding)
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

    #USFA with LSTM cumulants
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
            train_task_as_z=train_task_as_z,
            task_embed_type=task_embedding)
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

    # MSF with linear task embedding
    elif agent == "msf":
        # USFA + cumulants from FARM + Q-learning
        config = data_utils.merge_configs(
            dataclass_configs=[
                configs.ModularUSFAConfig(),
                configs.QAuxConfig(),
                configs.RewardConfig(),
                configs.FarmModelConfig(),
                configs.LangConfig(),
            ],
            dict_configs=default_config)

        return msf(
            config,
            env_spec,
            NetworkCls=nets.msf,
            predict_cumulants=True,
            learn_model=True,
            use_separate_eval=True,
            task_embedding=task_embedding)

    #MSF with conv cumulants
    elif agent == "conv_msf":
        # USFA + cumulants from FARM + Q-learning
        config = data_utils.merge_configs(
            dataclass_configs=[
                configs.ModularUSFAConfig(),
                configs.QAuxConfig(),
                configs.RewardConfig(),
                configs.FarmModelConfig(),
                configs.LangConfig(),
            ],
            dict_configs=default_config)
        config.cumulant_source = 'conv'

        return msf(
            config,
            env_spec,
            NetworkCls=nets.msf,
            predict_cumulants=True,
            learn_model=True,
            use_separate_eval=True,
            task_embedding=task_embedding)
    else:
        raise ValueError("Please specify a valid agent type")

    return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network

