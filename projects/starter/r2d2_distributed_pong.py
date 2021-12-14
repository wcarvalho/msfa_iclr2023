"""Runs DQN on bsuite locally."""

from absl import app
from absl import flags



import acme
from acme import wrappers
from acme.jax.networks import base

from functools import partial
from typing import Optional, Tuple


import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import gym

from agents import r2d2

# Bsuite flags
flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_boolean('local', False, 'local or distributed')

# flags.DEFINE_string('results_dir', 'results/atari', 'CSV results directory.')
# flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')

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

def make_environment(evaluation: bool = False,
                     level: str = 'PongNoFrameskip-v4') -> dm_env.Environment:
  env = gym.make(level, full_action_space=True)

  max_episode_len = 108_000 if evaluation else 50_000

  return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
  ])


def main(_):
  level = FLAGS.level
  environment_factory = lambda is_eval: make_environment(is_eval, level)

  if FLAGS.local:
    raise RuntimeError()
  else:
    program = r2d2.DistributedR2D2(
        environment_factory=environment_factory,
        network_factory=partial(r2d2.make_network,
          archCls=SimpleRecurrentQNetwork),
        config=r2d2.R2D2Config(**dict(
            min_replay_size=1,
            max_replay_size=100000,
            # batch_size=2,
            replay_period=40,
            trace_length=40,
            burn_in_length=40,
          )),
        num_actors=4,
        seed=1,
        max_number_of_steps=100000).build()

    # Launch experiment.
    lp.launch(program)


if __name__ == '__main__':
  app.run(main)