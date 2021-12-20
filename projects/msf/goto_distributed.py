"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments."""

import launchpad as lp

from absl import app
from absl import flags
import acme
import functools
from agents import td_agent

from projects.msf import helpers


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

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

  program = td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      builder=builder,
      config=config,
      seed=FLAGS.seed,
      num_actors=4).build()

  # Launch experiment.
  lp.launch(program)


if __name__ == '__main__':
  app.run(main)