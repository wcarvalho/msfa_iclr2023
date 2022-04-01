from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from utils import make_logger, gen_log_dir

from projects.common.environment_loop import EnvironmentLoop

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
    num_episodes: int=1_000):


  logger_fn = lambda : make_logger(
    wandb=False,
    log_dir=log_dir,
    label=loss_label)

  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
      logger_fn=logger_fn)

  kwargs={}
  if evaluate:
    kwargs['behavior_policy_constructor'] = functools.partial(td_agent.make_behavior_policy, evaluation=True)

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
  env_logger = make_logger(
    log_dir=log_dir,
    wandb=False,
    label='actor',
    steps_key="steps")

  loop = EnvironmentLoop(env, agent, logger=env_logger)
  loop.run(num_episodes)

