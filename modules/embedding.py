"""Modules for computing custom embeddings."""

import dataclasses

from acme.jax.networks import base
from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
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


class OneHotTask(hk.Module):
  """docstring for OneHotTask"""
  def __init__(self, size, dim, **kwargs):
    super(OneHotTask, self).__init__()
    self.size = size
    self.dim = dim
    self.embedder = hk.Embed(vocab_size=size, embed_dim=dim, **kwargs)
  
  def __call__(self, khot):

    each = self.embedder(jnp.arange(self.size))
    weighted = each*jnp.expand_dims(khot, axis=1)
    return weighted.sum(0)


class LanguageTaskEmbedder(hk.Module):
  """Module that embed words and then runs them through GRU."""
  def __init__(self, vocab_size, word_dim, task_dim, **kwargs):
    super(LanguageTaskEmbedder, self).__init__()
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.embedder = hk.Embed(vocab_size=vocab_size, embed_dim=word_dim, **kwargs)
    self.language_model = hk.GRU(task_dim)
  
  def __call__(self, x : jnp.ndarray):
    """Embed words, then run through GRU.
    
    Args:
        x (TYPE): B x N
    
    Returns:
        TYPE: Description
    """
    B, N = x.shape
    initial = self.language_model.initial_state(B) 
    words = self.embedder(x) # B x N x D
    words = jnp.transpose(words, (1,0,2))  # N x B x D
    sentence, _ = hk.static_unroll(self.language_model, words, initial)
    return sentence[-1] # embedding at end


