"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

To run:
  CUDA_VISIBLE_DEVICES=0 python projects/msf/goto_distributed.py --agent r2d1

"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import launchpad as lp

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
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 10, 'Number of actors.')

FLAGS = flags.FLAGS

def main(_):
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment(is_eval)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = helpers.load_agent_settings(FLAGS.agent, env_spec)

  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs)

  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
    )


  # -----------------------
  # loggers
  # -----------------------
  agent = str(FLAGS.agent)
  log_dir = gen_log_dir(
    base_dir="results/msf/distributed",
    agent=agent)
  logger_fn = lambda : make_logger(
        log_dir=log_dir, label=agent, asynchronous=True,
        )

  actor_logger_fn = lambda actor_id: make_logger(
                  log_dir=log_dir, label='actor',
                  save_data=actor_id == 0,
                  )
  evaluator_logger_fn = lambda : make_logger(
                  log_dir=log_dir, label='evaluator',
                  )

  # -----------------------
  # build program and run
  # -----------------------
  program = td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      builder=builder,
      logger_fn=logger_fn,
      actor_logger_fn=actor_logger_fn,
      evaluator_logger_fn=evaluator_logger_fn,
      config=config,
      workdir=log_dir,
      seed=FLAGS.seed,
      num_actors=FLAGS.num_actors).build()

  # Launch experiment.
  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == '__main__':
  app.run(main)
