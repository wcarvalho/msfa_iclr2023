"""Runs DQN on bsuite locally."""

import numpy as np
from functools import partial
from absl import app
from absl import flags
import acme
from acme.agents.jax import actor_core as actor_core_lib
from acme import specs
from acme import wrappers
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
import bsuite
import haiku as hk
import rlax
from acme.jax import utils


# Bsuite flags
flags.DEFINE_string('bsuite_id', 'catch/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', 'results/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')

FLAGS = flags.FLAGS

def make_network(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates networks used by the agent."""

  def actor_fn(obs, is_training=True, key=None):
    # is_training and key allows to utilize train/test dependant modules
    # like dropout.
    del is_training
    del key
    mlp = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP([50, 50, spec.actions.num_values])])
    return mlp(obs)

  policy = hk.without_apply_rng(hk.transform(actor_fn, apply_rng=True))

  # Create dummy observations to create network parameters.
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  network = networks_lib.FeedForwardNetwork(
      lambda key: policy.init(key, dummy_obs), policy.apply)

  return network

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
  network = make_network(environment_spec)

  # Create actor
  actor = dqn.DQN(
      environment_spec=environment_spec,
      network=network,
      batch_size=8,
  )


  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, actor)
  loop.run(num_episodes=100*environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)