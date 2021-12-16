from acme import types
from acme.jax import networks as networks_lib
from acme.jax.networks import embedding


class USFANetwork(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
        num_actions: int,
        lstm_size : int=256,
        hidden_size : int=256,
        ):
    super().__init__(name='r2d2_atari_network')
    self.conv = networks_lib.AtariTorso()
    self.memory = hk.LSTM(lstm_size)
    self.statefn = hk.nets.MLP(hidden_size)

    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[512])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: SuccessorFeatureInputs,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:

    last_action = inputs.action
    last_reward = inputs.reward
    observation = inputs.observation
    image = observation.image
    task = observation.task
    state = observation.state_features

    import ipdb; ipdb.set_trace()
    # -----------------------
    # 
    # -----------------------
    observation = self.conv(observation)
    memory_inputs = jnp.concatenate((observation, ))
    memory_outputs, memory_state = self.memory(observation, state)
    state = self.statefn(memory_outputs)






    embeddings = self._embed(inputs)  # [B, D+A+1]
    q_values = self._duelling_head(memory_outputs)
    return q_values, memory_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self.memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    memory_outputs, memory_states = hk.static_unroll(self.memory, embeddings, state)
    q_values = hk.BatchApply(self._duelling_head)(memory_outputs)  # [T, B, A]
    return q_values, memory_states