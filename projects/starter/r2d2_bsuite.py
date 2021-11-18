"""Runs DQN on bsuite locally.

python -m ipdb -c continue  projects/starter/r2d2_bsuite.py

JAX_DISABLE_JIT=1 python -m ipdb -c continue  projects/starter/r2d2_bsuite.py

"""

from absl import app
from absl import flags

import jax
import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple
from functools import partial

import acme
from acme.wrappers import observation_action_reward
from acme.agents.jax import actor_core as actor_core_lib
from acme import specs
from acme import wrappers
from acme.jax import networks as networks_lib
from acme.jax.networks import base
import bsuite
import haiku as hk
import rlax
from acme.jax import utils

from agents import r2d2
from agents.r2d2.agent import R2D2

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'catch/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', 'results/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')
flags.DEFINE_boolean('feedforward', False, 'Whether to do feedforward')

FLAGS = flags.FLAGS

class SimpleRecurrentQNetwork(hk.RNNCore):

  def __init__(self, num_actions: int, feedforward : bool=False):
    super().__init__(name='simple_r2d2_network')
    self._embed = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP([50, 50])])
    self._core = hk.LSTM(20)
    self._head = hk.nets.MLP([num_actions])
    self.feedforward = feedforward
    print("FEEDWORD", feedforward)

  def __call__(
      self,
      inputs: jnp.ndarray,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._embed(inputs)  # [B, D+A+1]
    if self.feedforward:
      new_state = state
      q_values = self._head(embeddings)
    else:
      core_outputs, new_state = self._core(embeddings, state)
      q_values = self._head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: jnp.ndarray,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    embeddings = hk.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    if self.feedforward:
      q_values = hk.BatchApply(self._head)(embeddings)  # [T, B, A]
      new_states = state
    else:
      core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
      q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return q_values, new_states

def main(_):

  # Create an environment and grab the spec.
  raw_environment = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )

  environment = wrappers.SinglePrecisionWrapper(raw_environment)
  spec = specs.make_environment_spec(environment)

  network = r2d2.make_network(spec, 
    partial(SimpleRecurrentQNetwork, feedforward=FLAGS.feedforward)
    )

  actor = r2d2.R2D2(
    spec=spec,
    network=network,
    config=r2d2.R2D2Config(**dict(
        min_replay_size=100,
        discount=.99,
        max_gradient_norm=40.0,
        burn_in_length=0,
        trace_length=10,
        batch_size=8,
        # max_replay_size=10000,
        # replay_period=4,
        # trace_length=4,
        # burn_in_length=4,
      )),
    seed=1
    )


  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, actor)
  loop.run(num_episodes=100*environment.bsuite_num_episodes) # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)