"""Runs IMPALA on bsuite locally."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.agents.jax import impala
from acme.jax import networks
import bsuite
import sonnet as snt

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')
FLAGS = flags.FLAGS


def make_network(action_spec: specs.DiscreteArray) -> snt.RNNCore:
  return snt.DeepRNN([
      snt.Flatten(),
      snt.nets.MLP([50, 50]),
      snt.LSTM(20),
      networks.PolicyValueHead(action_spec.num_values),
  ])


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
  network = make_network(environment_spec.actions)

  agent = impala.IMPALA(
      environment_spec=environment_spec,
      network=network,
      sequence_length=3,
      sequence_period=3,
  )

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)