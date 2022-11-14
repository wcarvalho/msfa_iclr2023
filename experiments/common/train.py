from absl import app
from absl import flags
import collections
import acme
from acme.jax import utils

import functools
import jax

from agents import td_agent
from utils import make_logger, gen_log_dir

from experiments.common.observers import LevelReturnObserver

from acme.environment_loop import EnvironmentLoop

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_string('env', 'fruitbot', 'which environment.')
flags.DEFINE_string('env_setting', '', 'which environment setting.')
flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('test', True, 'whether testing.')
flags.DEFINE_bool('evaluate', True, 'whether to use evaluation policy.')
flags.DEFINE_bool('init_only', False, 'whether to only init arch.')

# -----------------------
# wandb
# -----------------------
flags.DEFINE_bool('wandb', False, 'whether to log.')
flags.DEFINE_string('wandb_project', 'msf_sync', 'wand project.')
flags.DEFINE_string('wandb_entity', 'wcarvalho92', 'wandb entity')
flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('notes', '', 'notes for wandb.')


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
  CustomPreds = collections.namedtuple('CustomPreds', keys, defaults=(None,) * len(keys))

  # adds compatibility with pickling ++
  globals()['CustomPreds'] = CustomPreds
  import __main__
  setattr(__main__, CustomPreds.__name__, CustomPreds)
  CustomPreds.__module__ = "__main__"
  return CustomPreds


def run(env,
    env_spec,
    config,
    NetworkCls,
    NetKwargs,
    LossFn,
    LossFnKwargs,
    log_dir: str,
    loss_label: str="Loss",
    evaluate: bool=False,
    seed: int=1,
    num_episodes: int=1_000,
    log_every=30.0,
    log_with_key:str=None,
    observers=None,
    actor_label='actor',
    wandb_init_kwargs=None,
    init_only=False,
    pregenerate_named_tuple=True,
    **kwargs):

  # -----------------------
  # loggers + observers
  # -----------------------
  logger_fn = None
  env_logger = None
  if log_dir:
      use_wandb = True if wandb_init_kwargs is not None else False
      logger_fn = lambda: make_logger(
            log_dir=log_dir,
            label=loss_label,
            time_delta=log_every,
            log_with_key=log_with_key,
            wandb=use_wandb,
            max_number_of_steps=config.max_number_of_steps,
            asynchronous=True)

      env_logger = make_logger(
        log_dir=log_dir,
        label=actor_label,
        wandb=use_wandb,
        max_number_of_steps=config.max_number_of_steps,
        time_delta=log_every,
        log_with_key=log_with_key,
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
      learner_kwargs=dict(
        clear_sgd_cache_period=config.clear_sgd_cache_period,
        grad_period=config.grad_period)
      )

  kwargs=kwargs or {}
  if evaluate:
    kwargs['behavior_policy_constructor'] = functools.partial(td_agent.make_behavior_policy, evaluation=True)

  # -----------------------
  # prepare networks
  # -----------------------
  if pregenerate_named_tuple:
    PredCls = create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs)
    # insert into global namespace for pickling, etc.
    NetKwargs.update(PredCls=PredCls)
  if init_only:
    return

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

