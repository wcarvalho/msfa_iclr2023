from absl import app
from absl import flags
import collections
import acme
from acme.jax import utils

import functools
import jax

from agents import td_agent
from utils import make_logger, gen_log_dir

from projects.common.observers import LevelReturnObserver

from acme.environment_loop import EnvironmentLoop

def create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs, data_length = 2):
  """ Create dummy output and use that to define a NamedTuple that will be used to collect data. 
  Note: Assumes BasicArch will be used.
  
  Args:
      config (TYPE): Description
      env_spec (TYPE): Description
      NetworkCls (TYPE): Description
      NetKwargs (TYPE): Description
      data_length (int, optional): Description
  
  Returns:
      TYPE: Description
  """
  # -----------------------
  # preparate data
  # -----------------------
  dummy_obs = utils.tile_nested(
      utils.zeros_like(env_spec.observations), config.batch_size)
  dummy_obs_se = utils.tile_nested(dummy_obs, data_length)

  # -----------------------
  # prepare network
  # -----------------------
  dummy_networks = td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True)
  dummy_key = jax.random.PRNGKey(42)
  dummy_params = dummy_networks.init(dummy_key)
  dummy_state = dummy_networks.initial_state.apply(
    dummy_params, dummy_key, config.batch_size)
  dummy_output, _ = dummy_networks.unroll.apply(
    dummy_params,
    dummy_key,
    dummy_obs_se,
    dummy_state,
    dummy_key)

  keys = dummy_output._asdict().keys()
  return collections.namedtuple('Predictions', keys, defaults=(None,) * len(keys))

def run(env,
    env_spec,
    config,
    NetworkCls,
    NetKwargs,
    LossFn,
    LossFnKwargs,
    loss_label,
    log_dir: str,
    evaluate: bool=False,
    seed: int=1,
    num_episodes: int=1_000,
    log_every=5.0,
    observers=None,
    wandb_init_kwargs=None):

  # -----------------------
  # loggers + observers
  # -----------------------
  use_wandb = True if wandb_init_kwargs is not None else False
  logger_fn = lambda: make_logger(
        log_dir=log_dir,
        label=loss_label,
        time_delta=log_every,
        wandb=use_wandb,
        asynchronous=True)

  env_logger = make_logger(
    log_dir=log_dir,
    label='actor',
    wandb=use_wandb,
    time_delta=log_every,
    steps_key="steps")

  if wandb_init_kwargs is not None:
    import wandb
    wandb.init(**wandb_init_kwargs)

  observers = observers or [LevelReturnObserver()]
  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn,
      LossFnKwargs=LossFnKwargs,
      logger_fn=logger_fn,
      learner_kwargs=dict(clear_sgd_cache_period=config.clear_sgd_cache_period)
      )

  kwargs={}
  if evaluate:
    kwargs['behavior_policy_constructor'] = functools.partial(td_agent.make_behavior_policy, evaluation=True)

  # -----------------------
  # prepare networks
  # -----------------------
  PredCls = create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs)
  NetKwargs.update(PredCls=PredCls)

  agent = td_agent.TDAgent(
      env_spec,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True),
      builder=builder,
      workdir=log_dir,
      config=config,
      seed=seed,
      **kwargs,
      )

  # -----------------------
  # make env + run
  # -----------------------

  loop = EnvironmentLoop(
    env,
    agent,
    logger=env_logger,
    observers=observers or ())
  loop.run(num_episodes)

