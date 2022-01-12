import acme
import functools

from acme import wrappers
import dm_env

from envs.acme.goto_avoid import GoToAvoid
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper

from utils import ObservationRemapWrapper


from agents import td_agent
from projects.msf import networks as msf_networks


def make_environment(evaluation: bool = False,
                     tile_size=8,
                     path='.',
                     ) -> dm_env.Environment:
  """Loads environments."""
  if evaluation:
    obj2rew=dict(
        pan_plates={
            "pan" : 1,
            "plates" :1,
            "fork" : 0,
            "knife" : 0,
            },
        all={
            "pan" : 1,
            "plates" : 1,
            "fork" : 1,
            "knife" : 1,
            },
        mix1={
            "pan" : -1,
            "plates" : 1,
            "fork" : -1,
            "knife" : 1,
            },
        mix2={
            "pan" : -1,
            "plates" : 1,
            "fork" : 0,
            "knife" : 1,
            },
    )
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
    config = td_agent.R2D1Config(**default_config)

    NetworkCls=msf_networks.R2D2Network
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      lstm_size=256,
      hidden_size=128,
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)


  elif agent == "usfa": # Universal Successor Features
    config = td_agent.USFAConfig(**default_config)

    NetworkCls=msf_networks.USFANetwork
    state_dim = env_spec.observations.observation.state_features.shape[0]
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      state_dim=state_dim,
      lstm_size=256,
      hidden_size=128,
      nsamples=config.npolicies,
      variance=config.variance,
      )

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  elif agent == "usfa_reward":
    # Universal Successor Features which learns cumulants by predicting reward
    config = td_agent.USFAConfig(**default_config)

    NetworkCls=msf_networks.USFANetwork
    state_dim = env_spec.observations.observation.state_features.shape[0]
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      state_dim=state_dim,
      lstm_size=256,
      hidden_size=128,
      nsamples=config.npolicies,
      variance=config.variance,
      )

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  # elif agent == "msf": # Modular Successor Features
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs
