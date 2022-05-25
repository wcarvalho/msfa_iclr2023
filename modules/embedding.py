"""Modules for computing custom embeddings."""

import dataclasses

from acme.jax.networks import base
from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
Images = jnp.ndarray

import distrax

class OAREmbedding(hk.Module):
  """Module for embedding (observation, action, reward, task) inputs together."""

  def __init__(self, num_actions, concat=True, observation=True, **kwargs):
    super(OAREmbedding, self).__init__()
    self.num_actions = num_actions
    self.concat = concat
    self.observation = observation

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

  @property
  def out_dim(self):
    return self.dim


class Identity(hk.Module):
  """docstring for OneHotTask"""
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

class LanguageTaskEmbedder(hk.Module):
  """Module that embed words and then runs them through GRU."""
  def __init__(self,
      vocab_size: int,
      word_dim: int,
      sentence_dim: int,
      task_dim: int=None,
      initializer: str='TruncatedNormal',
      compress: str ='last',
      gates: int=None,
      gate_type: str='sample',
      tanh: bool=False,
      **kwargs):
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
    self.sentence_dim = sentence_dim
    self.language_model = hk.GRU(sentence_dim)

    if task_dim is None or task_dim == 0:
      self.task_dim = sentence_dim
      self.task_projection = lambda x:x
    else:
      self.task_dim = task_dim
      self.task_projection = hk.Linear(task_dim)

    self.gates = gates
    self.tanh = tanh
    self.gate_type = gate_type.lower()
    assert self.gate_type in ['round', 'sample', 'sigmoid']
    if self.gates is not None and self.gates > 0:
      self.gate = hk.Linear(gates)
  
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
      task = sentence[-1] # embedding at end
    elif self.compress == "sum":
      task = sentence.sum(0)
    else:
      raise NotImplementedError(self.compress)

    # [B, D]
    task_proj = self.task_projection(task)

    if self.tanh:
      task_proj = jax.nn.tanh(task_proj)

    if self.gates is not None and self.gates > 0:
      # [B, G]
      gate = jax.nn.sigmoid(self.gate(task))
      if self.gate_type == 'sample':
        gate = st_bernoulli(gate, key=hk.next_rng_key())
      elif self.gate_type == 'round':
        gate = st_round(gate)

      # [B, D] --> [B, G, D/G]
      task_proj = jnp.stack(jnp.split(task_proj, self.gates, axis=-1), axis=1) 
      # [B, G, D/G] * [B, G, 1]
      task_proj = task_proj*jnp.expand_dims(gate, 2)
      task_proj = task_proj.reshape(B, -1)

    return task_proj


  @property
  def out_dim(self):
    return self.task_dim
