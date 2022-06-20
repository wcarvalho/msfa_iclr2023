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
  def __init__(self, hidden_dim=128, out_dim=6, num_tasks=None, structured=False, **kwargs):
    super(LinearTaskEmbedding, self).__init__()
    self.hidden_dim = hidden_dim
    self.structured = structured
    self._out_dim = out_dim
    if structured:
      assert num_tasks is not None
      self.layer2_dim = out_dim//num_tasks
    else:
      self.layer2_dim = out_dim

    if hidden_dim > 0:
      self.layer1 = hk.Linear(hidden_dim,
        with_bias=False, 
        w_init=hk.initializers.RandomNormal(
            stddev=1., mean=0.))

      self.layer2 = hk.Linear(self.layer2_dim)
    else:
      self.layer1 = lambda x:x
      self.layer2 = hk.Linear(out_dim,
        with_bias=False, 
        w_init=hk.initializers.RandomNormal(
            stddev=1., mean=0.))

  def __call__(self, x):
    """Summary
    
    Args:
        x (TYPE): B x D
    """
    def apply_net(x_):
      y = self.layer1(x_)
      if self.hidden_dim > 0:
        y = jax.nn.relu(y)
      return self.layer2(y)

    if self.structured:
      # [D, D]
      block_id = jnp.identity(x.shape[-1])
      mul = jax.vmap(jnp.multiply, in_axes=(None, 0), out_axes=1)
      # [B, D, D]
      struct_x = mul(x, block_id)

      z = hk.BatchApply(apply_net)(struct_x)

      # [B, D_out]
      # combine all tasks into one embedding
      z = z.reshape(x.shape[0], -1)
    else:
      z = apply_net(x)

    return z 

  @property
  def out_dim(self):
    return self._out_dim




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

class LanguageTaskEmbedder(hk.Module):
  """Module that embed words and then runs them through GRU. The Token`0` is treated as padding and masked out."""
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
      relu: bool=False,
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
    self.relu = relu
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

    # -----------------------
    # embed words + mask
    # -----------------------
    words = self.embedder(x) # B x N x D
    mask = (x > 0).astype(words.dtype)
    words = words*jnp.expand_dims(mask, axis=-1)

    # -----------------------
    # pass through GRU
    # -----------------------
    initial = self.language_model.initial_state(B)
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
    if self.relu:
      task_proj = jax.nn.relu(task_proj)

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


def embed_position(factors: jnp.ndarray, size: int):
  """Summary
  
  Args:
      factors (jnp.ndarray): B x N x ...
      size (int): size of embedding
  
  Returns:
      TYPE: Description
  """
  N = factors.shape[1]
  embedder = hk.Embed(
    vocab_size=N,
    embed_dim=size)
  embeddings = embedder(jnp.arange(N)) # N x D
  concat = lambda a,b: jnp.concatenate((a, b), axis=-1)
  factors = jax.vmap(concat, in_axes=(0, None), out_axes=0)(
    factors, embeddings)
  return factors
