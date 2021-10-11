"""Runs IMPALA on bsuite locally."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.agents.jax import dqn
from acme.jax import networks
import bsuite
import haiku as hk

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'catch/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', 'results/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')
FLAGS = flags.FLAGS


def main(_):
  # Create an environment and grab the spec.
  raw_environment = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
  environment = wrappers.SinglePrecisionWrapper(raw_environment)
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize.
  network = hk.DeepRNN([
      hk.Flatten(),
      hk.nets.MLP([50, 50]),
      hk.LSTM(20),
      networks.PolicyValueHead(environment_spec.actions.num_values),
  ])

  agent = dqn.DQN(
      environment_spec=environment_spec,
      network=network,
  )

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)