import acme
import functools

from acme import wrappers
import dm_env

from envs.acme.goto_avoid import GoToAvoid
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper

from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import aux_tasks
from agents.td_agent import losses
from projects.msf import nets
from projects.msf import networks as msf_networks
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
  default_config = dict(
    # network
    discount=0.99,
    target_update_period=2500,

    # Learner options
    trace_length=40,
    learning_rate=5e-5,
    max_number_of_steps=50_000_000, # 5M takes 1hr

    # How many gradient updates to perform per learner step.
    num_sgd_steps_per_step=4,

    # Replay options
    batch_size=32,
    min_replay_size=10_000,
    max_replay_size=250_000,
    num_parallel_calls=1,
    prefetch_size=0,
    )
  default_config.update(config_kwargs or {})

  if agent == "r2d1": # Recurrent DQN
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.make_r2d1 # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'r2d1'

  elif agent == "usfa": # Universal Successor Features
    state_dim = env_spec.observations.observation.state_features.shape[0]

    config = configs.USFAConfig(**default_config)
    config.state_dim = state_dim

    NetworkCls=nets.make_usfa # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)


    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'usfa'

  elif agent == "r2d1_farm":

    config = configs.R2D1FarmConfig(**default_config)
    NetworkCls=nets.make_r2d1_farm # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'r2d1'


  elif agent == "usfa_farm_flat":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFAFarmConfig(**default_config)

    NetworkCls =  nets.usfa_farm_flat # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'usfa'

  elif agent == "usfa_reward":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFARewardConfig(**default_config)

    NetworkCls =  nets.make_usfa_reward # default: 2.1M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulant=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=functools.partial(aux_tasks.cumulant_from_reward,
        coeff=config.reward_coeff,  # coefficient for loss
        loss=config.reward_loss))   # type of loss for reward

    loss_label = 'usfa'

  elif agent == "usfa_reward_vae":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFARewardConfig(**default_config)

    NetworkCls =  nets.make_usfa_reward_vae # default: 2.1M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    aux_tasks = [
      functools.partial(aux_tasks.cumulant_from_reward,
        coeff=config.reward_coeff,  # coefficient for loss
        loss=config.reward_loss)
    ]
    LossFnKwargs.update(
      extract_cumulant=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=aux_tasks)   # type of loss for reward

    loss_label = 'usfa'


  # ======================================================
  # Unchecked
  # ======================================================

  # elif agent == "usfa_farm":
  #   # Universal Successor Features which learns cumulants by predicting reward
  #   config = configs.USFAConfig(**default_config)

  #   NetworkCls =  msf_networks.UsfaFarmMixture
  #   state_dim = env_spec.observations.observation.state_features.shape[0]
  #   NetKwargs=dict(
  #     num_actions=env_spec.actions.num_values,
  #     state_dim=state_dim,
  #     lstm_size=128,
  #     hidden_size=128,
  #     nsamples=config.npolicies,
  #     variance=config.variance,
  #     )

  #   LossFn = td_agent.USFALearning

  #   LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
  #   # LossFnKwargs.update(
  #   #   extract_cumulant=losses.cumulants_from_preds,
  #   #   # auxilliary task as argument
  #   #   aux_tasks=functools.partial(
  #   #     aux_tasks.cumulant_from_reward,
  #   #       coeff=config.reward_coeff,  # coefficient for loss
  #   #       loss=config.reward_loss))   # type of loss for reward

  #   loss_label = 'usfa'
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label
