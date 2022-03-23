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

def make_environment_sanity_check(simple: bool = True):
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

    wrapper_list = [
        functools.partial(ObservationRemapWrapper,
                          remap=dict(mission='task')),
        wrappers.ObservationActionRewardWrapper,
        wrappers.SinglePrecisionWrapper,
    ]
    return wrappers.wrap_all(env, wrapper_list)


def load_agent_settings_sanity_check(env_spec, config_kwargs=None):
    default_config = dict()
    default_config.update(config_kwargs or {})

    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs

