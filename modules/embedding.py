"""Modules for computing custom embeddings."""

import dataclasses

from acme.jax.networks import base
from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp

Images = jnp.ndarray


@dataclasses.dataclass
class OAREmbedding(hk.Module):
  """Module for embedding (observation, action, reward, task) inputs together."""

  num_actions: int
  concat: bool=True
  observation: bool=True


  def __call__(self,
    inputs: observation_action_reward.OAR, obs: jnp.array=None) -> jnp.ndarray:
    """Embed each of the (observation, action, reward) inputs & concatenate."""

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs.action, num_classes=self.num_actions)  # [T?, B, A]

    # Map rewards -> [-1, 1].
    reward = jnp.tanh(inputs.reward)

    # Add dummy trailing dimensions to rewards if necessary.
    while reward.ndim < action.ndim:
      reward = jnp.expand_dims(reward, axis=-1)

    # Concatenate on final dimension.
    items = [action, reward]

    if self.observation:
      assert obs is not None, "provide observation"
      items.append(obs)

    if self.concat:
      items = jnp.concatenate(items, axis=-1)  # [T?, B, D+A+1]

    return items
