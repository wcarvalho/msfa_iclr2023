"""Runs DQN on bsuite locally."""

from absl import app
from absl import flags

import jax
import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple


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

from agents.r2d2.agent import R2D2
from agents.r2d2.networks import R2D2Network

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'catch/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', 'results/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')

FLAGS = flags.FLAGS

class SimpleRecurrentQNetwork(hk.RNNCore):

  def __init__(self, num_actions: int):
    super().__init__(name='r2d2_atari_network')
    self._embed = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP([50, 50])])
    self._core = hk.LSTM(20)
    self._head = hk.nets.MLP([num_actions])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: jnp.ndarray,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._embed(inputs)  # [B, D+A+1]
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


  # Create pure functions
  def forward_fn(x : jnp.ndarray, s : hk.LSTMState):
    model = SimpleRecurrentQNetwork(spec.actions.num_values)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = SimpleRecurrentQNetwork(spec.actions.num_values)
    return model.initial_state(batch_size)

  def unroll_fn(inputs : jnp.ndarray, state : hk.LSTMState):
    model = SimpleRecurrentQNetwork(spec.actions.num_values)
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
    # for time
    dummy_obs = utils.add_batch_dim(dummy_obs)
    # dummy_obs = add_time(dummy_obs)
    # TODO: params are not returned, only initial_params
    # so currently don't support learning params for intialization
    params = initial_state_fn_hk.init(key)
    batch_size = 1
    initial_state = initial_state_fn_hk.apply(params, batch_size)
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
      min_replay_size=100,
      max_replay_size=10000,
      # batch_size=2,
      replay_period=4,
      trace_length=4,
      burn_in_length=4,
  )


  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, actor)
  loop.run(num_episodes=100*environment.bsuite_num_episodes) # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)