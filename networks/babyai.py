from acme.jax import networks as networks_lib
from acme.jax.networks import embedding

# class BabyAITorso(base.Module):
#   """Simple convolutional stack commonly used for Atari."""

#   def __init__(self, tile_size=8):
#     super().__init__(name='atari_torso')
#     self._network = hk.Sequential([
#         hk.Conv2D(output_channels=128,
#             kernel_spae=[tile_size, tile_size],
#             stride=tile_size,
#             padding=0),
#         hk.Conv2D(output_channels=128,
#             kernel_spae=[tile_size, tile_size],
#             stride=1,
#             padding=1),
#         jax.nn.relu,
#         hk.Conv2D(output_channels=64,
#             kernel_spae=[4, 4],
#             stride=2),
#         jax.nn.relu,
#         hk.Conv2D(output_channels=64,
#             kernel_spae=[3, 3],
#             stride=1),
#         jax.nn.relu
#     ])

#   def __call__(self, inputs: Images) -> jnp.ndarray:
#     inputs_rank = jnp.ndim(inputs)
#     batched_inputs = inputs_rank == 4
#     if inputs_rank < 3 or inputs_rank > 4:
#       raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

#     outputs = self._network(inputs)

#     if batched_inputs:
#       return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
#     return jnp.reshape(outputs, [-1])  # [D]


class R2D2BabyAINetwork(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
        num_actions: int,
        tile_size : int=8,
        lstm_size : int=512,
        duelling_size : int=512,
        ):
    super().__init__(name='r2d2_atari_network')
    self.conv = networks_lib.AtariTorso()

    self._embed = embedding.OAREmbedding(, num_actions)

    self._core = hk.LSTM(lstm_size)
    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[512])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._embed(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._duelling_head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._duelling_head)(core_outputs)  # [T, B, A]
    return q_values, new_states