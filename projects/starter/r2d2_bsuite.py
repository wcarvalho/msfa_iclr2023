"""Runs DQN on bsuite locally."""

import numpy as np
from functools import partial
from absl import app
from absl import flags
import acme
from acme.agents.jax import actor_core as actor_core_lib
from acme import specs
from acme import wrappers
from acme.jax import networks as networks_lib
import bsuite
import haiku as hk
import rlax
from acme.jax import utils

from agents.r2d2.agent import R2D2
from agents.r2d2.networks import R2D2Network

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
  spec = specs.make_environment_spec(environment)


  # Create pure functions
  def forward_fn(x, s):
    model = networks_lib.R2D2AtariNetwork(spec.actions.num_values)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = networks_lib.R2D2AtariNetwork(spec.actions.num_values)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
    model = networks_lib.R2D2AtariNetwork(spec.actions.num_values)
    return model.unroll(inputs, state)

  # We pass pure, Haiku-agnostic functions to the agent.
  forward_fn_hk = hk.without_apply_rng(hk.transform(
      forward_fn,
      apply_rng=True))
  unroll_fn_hk = hk.without_apply_rng(hk.transform(
      unroll_fn,
      apply_rng=True))
  initial_state_fn_hk = hk.without_apply_rng(hk.transform(
      initial_state_fn,
      apply_rng=True))

  def init(key):
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
    # TODO: params are not returned, only initial_params
    # so currently don't support learning params for intialization
    params = initial_state_fn_hk.init(key)
    initial_state = initial_state_fn_hk.apply(params)
    key, key_initial_state = jax.random.split(key)
    initial_params = unroll_fn_hk.init(key, dummy_obs, initial_state)
    return initial_params


  network = R2D2Network(
      init=init, # create params
      apply=forward_fn_hk.apply, # call
      unroll=unroll_fn_hk.apply, # unroll
      initial_state=initial_state_fn_hk.apply, # initial_state
  )

  # Create actor
  actor = R2D2(
      environment_spec=spec,
      network=network,
      burn_in_length=4,
      trace_length=4,
      replay_period=4,
  )


  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, actor)
  loop.run(num_episodes=100*environment.bsuite_num_episodes) # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)