import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

import os.path

from absl import app
from absl import flags
import acme
from acme.utils import paths
import functools

from agents import td_agent
from utils import make_logger, gen_log_dir
from utils import data as data_utils

from projects.common.environment_loop import EnvironmentLoop

def build_common_program(
  environment_factory,
  env_spec,
  log_dir : str,
  num_actors : int,
  config,
  save_config_dict,
  NetworkCls,
  NetKwargs,
  LossFn,
  LossFnKwargs,
  loss_label,
  wandb_init_kwargs=None,
  log_every=30.0, # how often to log
  ):

  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      eval_network=True)

  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
    )

  # -----------------------
  # loggers
  # -----------------------
  use_wandb = True if wandb_init_kwargs is not None else False
  logger_fn = lambda : make_logger(
        log_dir=log_dir,
        label=loss_label,
        wandb=use_wandb,
        asynchronous=True)

  actor_logger_fn = lambda actor_id: make_logger(
                  log_dir=log_dir, label='actor',
                  wandb=use_wandb,
                  save_data=actor_id == 0,
                  steps_key="actor_steps",
                  )
  evaluator_logger_fn = lambda : make_logger(
                  log_dir=log_dir, label='evaluator',
                  wandb=use_wandb,
                  steps_key="evaluator_steps",
                  )

  # -----------------------
  # wandb setup
  # -----------------------
  if wandb_init_kwargs:
    # add config to wandb
    wandb_config = wandb_init_kwargs.get("config", {})
    wandb_config.update(save_config_dict)
    wandb_init_kwargs['config'] = wandb_config

  def wandb_wrap_logger(_logger_fn):
    """This will start wandb inside each child process"""
    def make_logger(*args, **kwargs):
      import wandb
      wandb.init(**wandb_init_kwargs)
      return _logger_fn(*args, **kwargs)
    return make_logger

  if wandb_init_kwargs is not None:
    import wandb
    wandb.init(**wandb_init_kwargs)

    logger_fn = wandb_wrap_logger(logger_fn)
    actor_logger_fn = wandb_wrap_logger(actor_logger_fn)
    evaluator_logger_fn = wandb_wrap_logger(evaluator_logger_fn)

  # -----------------------
  # save config
  # -----------------------
  paths.process_path(log_dir)
  config_path = os.path.join(log_dir, 'config.json')
  data_utils.save_dict(
    file=config_path,
    dictionary=save_config_dict)

  # -----------------------
  # build program
  # -----------------------
  return td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      builder=builder,
      logger_fn=logger_fn,
      actor_logger_fn=actor_logger_fn,
      evaluator_logger_fn=evaluator_logger_fn,
      EnvLoopCls=EnvironmentLoop,
      config=config,
      # workdir=log_dir,
      seed=config.seed,
      num_actors=num_actors,
      max_number_of_steps=config.max_number_of_steps,
      log_every=log_every).build()