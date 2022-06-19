"""Modules for computing custom embeddings."""

import dataclasses

from acme.jax.networks import base
from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
Images = jnp.ndarray



class OAREmbedding(hk.Module):
  """Module for embedding (observation, action, reward, task) inputs together."""

  def __init__(self, num_actions, concat=True, observation=True, **kwargs):
    super(OAREmbedding, self).__init__()
    self.num_actions = num_actions
    self.concat = concat
    self.observation = observation

  def __call__(self,
    inputs: observation_action_reward.OAR, obs: jnp.array=None, extras=None) -> jnp.ndarray:
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
    if extras:
      items = items + extras

    if self.observation:
      assert obs is not None, "provide observation"
      items.append(obs)

    if self.concat:
      items = jnp.concatenate(items, axis=-1)  # [T?, B, D+A+1]

    return items


class BabyAIEmbedding(OAREmbedding):
  """docstring for SymbolicBabyAIEmbedding"""
  def __init__(self, *args, symbolic=False, **kwargs):
    super(BabyAIEmbedding, self).__init__(*args, **kwargs)
    self.symbolic = symbolic

  def __call__(self, inputs: observation_action_reward.OAR, obs: jnp.array=None):

    if self.symbolic:
      direction = inputs.observation.direction
      rank = len(direction.shape)
      direction = jax.nn.one_hot(direction.astype(jnp.int32), num_classes=5)
      extras = [direction]
    else:
      extras = None
    return super(BabyAIEmbedding, self).__call__(
      inputs=inputs,
      obs=obs,
      extras=extras)



class LinearTaskEmbedding(hk.Module):
  """docstring for LinearTaskEmbedding"""
  def __init__(self, dim, **kwargs):
    super(LinearTaskEmbedding, self).__init__()
    self.dim = dim
    self.layer = hk.Linear(dim,
      with_bias=False, 
      w_init=hk.initializers.RandomNormal(
          stddev=1., mean=0.))

  def __call__(self, x):
    return self.layer(x)

<<<<<<< HEAD
  @property
  def out_dim(self):
    return self.dim




class Identity(hk.Module):
  def __init__(self, dim):
    super(Identity, self).__init__()
    self.dim = dim
  
  def __call__(self, x):
    return x

  @property
  def out_dim(self):
    return self.dim

def st_bernoulli(x, key):
  """Straight-through bernoulli sample"""
  zero = x - jax.lax.stop_gradient(x)
  x = distrax.Bernoulli(probs=x).sample(seed=key)

  return zero + jax.lax.stop_gradient(x)

def st_round(x):
  """Straight-through bernoulli sample"""
  zero = x - jax.lax.stop_gradient(x)
  x = jnp.round(x)
  return zero + jax.lax.stop_gradient(x)
=======
>>>>>>> parent of d34fcbe (maybe we can merge this?)

class LanguageTaskEmbedder(hk.Module):
  """Module that embed words and then runs them through GRU."""
  def __init__(self, vocab_size, word_dim, task_dim,
    initializer='TruncatedNormal', compress='last', **kwargs):
    super(LanguageTaskEmbedder, self).__init__()
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.compress = compress
    initializer = getattr(hk.initializers, initializer)()
    self.embedder = hk.Embed(
      vocab_size=vocab_size,
      embed_dim=word_dim,
      w_init=initializer,
      **kwargs)
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
    if self.compress == "last":
      return sentence[-1] # embedding at end
    elif self.compress == "sum":
      return sentence.sum(0)
    else:
      raise NotImplementedError(self.compress)


