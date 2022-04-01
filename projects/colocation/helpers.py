import functools
from acme import wrappers
from utils import ObservationRemapWrapper
from agents import td_agent
from envs.acme.multiroom_goto import MultiroomGoto
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
import tensorflow as tf
import dm_env


# -----------------------
# specific to these set of experiments
# -----------------------
from projects.colocation import nets
from projects.colocation import configs

def make_environment_sanity_check( evaluation: bool = False, simple: bool = True, agent='r2d1'):
    if simple:
        objs = [{'pan': 1}, {'tomato': 1}, {'knife':1}]
    else:
        objs = [{'pan': 1,'pot':1,'stove':1}, {'tomato': 1,'lettuce':1}, {'knife':1,'apple':1}]
    env = MultiroomGoto(
        agent_view_size=5,
        objectlists={'level':objs},
        pickup_required=False,
        tile_size=10,
        epsilon=0.0,
        room_size=5,
        doors_start_open=True,
        stop_when_gone=True,
        wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=10)]
    )


    if agent=='usfa':
        print("USFA Agent baybee!!!")
        wrapper_list = [
            functools.partial(ObservationRemapWrapper,
                              remap=dict(mission='task', pickup='state_features')),
            wrappers.ObservationActionRewardWrapper,
            wrappers.SinglePrecisionWrapper,
        ]
    else:
        wrapper_list = [
            functools.partial(ObservationRemapWrapper,
                              remap=dict(mission='task')),
            wrappers.ObservationActionRewardWrapper,
            wrappers.SinglePrecisionWrapper,
        ]
    return wrappers.wrap_all(env, wrapper_list)

def load_agent_settings_sanity_check(env_spec, config_kwargs=None, agent = "r2d1"):
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
        config = configs.USFAConfig(**default_config)  # for convenience since has var

        NetworkCls = nets.r2d1_noise  # default: 2M params
        NetKwargs = dict(config=config, env_spec=env_spec)
        LossFn = td_agent.R2D2Learning
        LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
        loss_label = 'r2d1'
        eval_network = config.eval_network
    elif agent=='usfa':
        state_dim = env_spec.observations.observation.state_features.shape[0]

        config = configs.USFAConfig(**default_config)
        config.state_dim = state_dim

        NetworkCls = nets.usfa  # default: 2M params
        NetKwargs = dict(config=config, env_spec=env_spec)

        LossFn = td_agent.USFALearning
        LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

        loss_label = 'usfa'
        eval_network = config.eval_network
    else:
        raise ValueError("Please specify a valid agent type")

    return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network



