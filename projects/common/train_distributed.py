import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

import os.path

from absl import app
from absl import flags
import acme
from acme.utils import paths
from acme.jax.layouts import distributed_layout
import functools

from agents import td_agent
from utils import make_logger, gen_log_dir
from utils import data as data_utils
# from projects.common.environment_loop import EnvironmentLoop
from projects.common.observers import LevelReturnObserver
from projects.common.train import create_net_prediction_tuple

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
  colocate_learner_replay=True,
  observers=None,
  ):

  # -----------------------
  # prepare networks
  # -----------------------
  PredCls = create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs)
  NetKwargs.update(PredCls=PredCls)

  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      eval_network=True)

  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn,
      LossFnKwargs=LossFnKwargs,
      learner_kwargs=dict(clear_sgd_cache_period=config.clear_sgd_cache_period),
    )

  # -----------------------
  # loggers + observers
  # -----------------------
  use_wandb = True if wandb_init_kwargs is not None else False
  logger_fn = lambda : make_logger(
        log_dir=log_dir,
        label=loss_label,
        time_delta=log_every,
        wandb=use_wandb,
        asynchronous=True)

  actor_logger_fn = lambda actor_id: make_logger(
                  log_dir=log_dir, label='actor',
                  time_delta=log_every,
                  wandb=use_wandb,
                  save_data=actor_id == 0,
                  steps_key="actor_steps",
                  )
  evaluator_logger_fn = lambda label, steps_key: make_logger(
                  log_dir=log_dir, label='evaluator',
                  time_delta=log_every,
                  wandb=use_wandb,
                  steps_key="evaluator_steps",
                  )

  observers = observers or [LevelReturnObserver()]
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

  ckpt_config= distributed_layout.CheckpointingConfig(
    directory=log_dir)

  # -----------------------
  # build program
  # -----------------------
  print("multithreading_colocate_learner_and_reverb", multithreading_colocate_learner_and_reverb)
  import ipdb; ipdb.set_trace()
  return td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      builder=builder,
      logger_fn=logger_fn,
      actor_logger_fn=actor_logger_fn,
      evaluator_logger_fn=evaluator_logger_fn,
      # EnvLoopCls=EnvironmentLoop,
      config=config,
      checkpointing_config=ckpt_config,
      seed=config.seed,
      num_actors=num_actors,
      max_number_of_steps=config.max_number_of_steps,
      log_every=log_every,
      observers=observers,
      multithreading_colocate_learner_and_reverb=colocate_learner_replay).build()