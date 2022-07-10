from acme.wrappers import observation_action_reward
from acme.jax.networks import base


from typing import NamedTuple, Optional, Tuple
import haiku as hk
import jax
import jax.numpy as jnp


from archs.recurrent_q_network import RecurrentQNetwork
from modules.vision import AtariVisionTorso

def convert_floats(inputs):
  return jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

def get_image_from_inputs(inputs : observation_action_reward.OAR):
  return inputs.observation.image/255.0

class USFAPreds(NamedTuple):
  q: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  z: jnp.ndarray  # policy vector
  w: jnp.ndarray  # task vector

class UniveralSuccessorFeatureApproximator(RecurrentQNetwork):

  """Simple Recurrent Q Network Network.
  """
  
  def __init__(self, num_actions: int, rnn_size: int=512):
    super().__init__(
      num_actions=num_actions,
      rnn_size=rnn_size,
      name='usfa_network')
    self._embed = AtariVisionTorso()
    self._core = hk.LSTM(rnn_size)
    self._head = hk.nets.MLP([num_actions])

  def __call__(
      self,
      inputs: observation_action_reward,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    inputs = convert_floats(inputs)
    image = get_image_from_inputs(inputs)

    embeddings = self._embed(image)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._head(core_outputs)
    return Predictions(q=q_values), new_state

  def evaluate(
      self,
      inputs: observation_action_reward,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    inputs = convert_floats(inputs)
    image = get_image_from_inputs(inputs)

    embeddings = self._embed(image)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._head(core_outputs)
    return Predictions(q=q_values), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    inputs = convert_floats(inputs)
    image = get_image_from_inputs(inputs)

    embeddings = hk.BatchApply(self._embed)(image)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return Predictions(q=q_values), new_states
