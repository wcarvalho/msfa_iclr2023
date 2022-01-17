"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments."""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.msf import helpers
from projects.msf.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
  env = helpers.make_environment()
  env_spec = acme.make_environment_spec(env)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label = helpers.load_agent_settings(FLAGS.agent, env_spec)

  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/msf/local",
    agent=FLAGS.agent,
    seed=config.seed)
  logger_fn = lambda : make_logger(
        log_dir=log_dir, label=loss_label)


  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
      logger_fn=logger_fn)
  agent = td_agent.TDAgent(
      env_spec,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs),
      builder=builder,
      workdir=log_dir,
      config=config,
      seed=FLAGS.seed,
      )

  # -----------------------
  # make env + run
  # -----------------------
  env_logger = make_logger(
    log_dir=log_dir,
    label='actor',
    steps_key="steps")

  loop = EnvironmentLoop(env, agent, logger=env_logger)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
