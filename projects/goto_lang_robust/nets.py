from typing import Callable, Optional, Tuple, NamedTuple
import functools

from acme.jax.networks import duelling
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp

from agents import td_agent
from agents.td_agent.types import Predictions
from modules.basic_archs import BasicRecurrent
from modules.embedding import OAREmbedding, LanguageTaskEmbedder
from modules.ensembles import QEnsembleInputs, QEnsembleHead
from modules.vision import AtariVisionTorso, BabyAIVisionTorso
from modules.usfa import ConcatFlatStatePolicy

from utils import data as data_utils

class DuellingMLP(duelling.DuellingMLP):
  def __call__(self, *args, **kwargs):
    kwargs.pop("key", None)
    q = super().__call__(*args, **kwargs)
    return Predictions(q=q)

# ======================================================
# Processing functions
# ======================================================

def convert_floats(inputs):
  return jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

def get_image_from_inputs(inputs : observation_action_reward.OAR):
  return inputs.observation.image/255.0

def embed_task(inputs, task_embedder):
  """Convert task to ints, batchapply if necessary, and run through embedding function."""
  task = inputs.observation.task
  has_time = len(task.shape) == 3
  batchfn = hk.BatchApply if has_time else lambda x:x
  return batchfn(task_embedder)(task.astype(jnp.uint8))

def prediction_prep_fn(inputs, memory_out, task_embedder, **kwargs):
  """
  Concat task with memory output.
  """
  task = embed_task(inputs, task_embedder)
  return jnp.concatenate((memory_out, task), axis=-1)


class GatedAttention(hk.Module):
  """Chaplot.
  """
  def __init__(self):
    super(GatedAttention, self).__init__()

  def __call__(
      self,
      task: jnp.ndarray, # [B, D]
      image: jnp.ndarray, # [B, H, W, C]
  ) -> jnp.ndarray:

    dim = image.shape[-1]

    # ======================================================
    # compute coefficients
    # ======================================================
    coefficients = hk.Linear(dim)(task)
    coefficients = jnp.expand_dims(coefficients, (-3,-2)) # [B, H, W, D]
    coefficients = jax.nn.sigmoid(coefficients)

    # ======================================================
    # attend + projections
    # ======================================================
    image = image*coefficients
    image = hk.Conv2D(dim, [1, 1], 1)(image)
    return image

# ======================================================
# Networks
# ======================================================

def build_vision_net(config):
  if config.vision_torso == 'atari':
    vision = AtariVisionTorso(conv_dim=0, out_dim=config.vision_size)
  elif config.vision_torso == 'babyai':
    vision = BabyAIVisionTorso(
      batch_norm=config.vision_batch_norm,
      )
  else:
    raise NotImplementedError
  return vision

def build_task_embedder(config):
  return LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        task_dim=config.word_dim,
        initializer=config.word_initializer,
        compress=config.word_compress)

def r2d1(config, env_spec):
  num_actions = env_spec.actions.num_values

  vision = build_vision_net(config)
  task_embedder = build_task_embedder(config)

  embedder = OAREmbedding(num_actions=num_actions)
  def task_in_memory_prep_fn(inputs, obs):
    task = embed_task(inputs, task_embedder)
    oar = embedder(inputs, obs)
    return jnp.concatenate((oar, task), axis=-1)

  if config.task_in_memory:
    memory_prep_fn=task_in_memory_prep_fn
    pred_prep_fn=None # just use memory output
  else:
    memory_prep_fn=embedder
    pred_prep_fn=functools.partial(prediction_prep_fn,
      task_embedder=task_embedder)

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vision,
    memory_prep_fn=memory_prep_fn,
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=pred_prep_fn,
    prediction=DuellingMLP(num_actions,
      hidden_sizes=[config.out_hidden_size])
  )

def r2d1_noise(config, env_spec):
  num_actions = env_spec.actions.num_values
  variance = config.variance

  task_embedder = build_task_embedder(config)
  embedder = OAREmbedding(num_actions=num_actions)
  def noisy_task_in_memory_prep_fn(inputs, obs):
    """
    1. embed task
    2. Concat [task + noise] with memory output.
    """
    task = embed_task(inputs, task_embedder)
    noise = jax.random.normal(hk.next_rng_key(), task.shape)
    task =  task + jnp.sqrt(variance) * noise
    oar = embedder(inputs, obs)
    return jnp.concatenate((oar, task), axis=-1)

  if config.task_in_memory:
    memory_prep_fn=noisy_task_in_memory_prep_fn
    pred_prep_fn=None # just use memory output

    if config.eval_network:
      raise NotImplementedError("Need settings that create jax function which doesn't sample before putting into RNN.")
    else:
      evaluation_prep_fn = None # just use memory output
  else:
    raise NotImplementedError

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=build_vision_net(config),
    memory_prep_fn=memory_prep_fn,
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=pred_prep_fn,
    evaluation_prep_fn=evaluation_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )


def r2d1_noise_ensemble(config, env_spec):
  num_actions = env_spec.actions.num_values

  vision = build_vision_net(config)

  def ensemble_prep_fn(inputs, memory_out, task_embedder, **kwargs):
    """
    1. embed task
    2. Create inputs where task is language embedding
    """
    task = embed_task(inputs, task_embedder)
    return QEnsembleInputs(
      w=task,
      memory_out=memory_out,
      )

  if config.task_in_memory:
    raise NotImplementedError("Need way to deal with N samples going into RNN....")
    memory_prep_fn=noisy_task_in_memory_prep_fn
    pred_prep_fn=None # just use memory output
    evaluation_prep_fn = None # just use memory output

  else:
    raise NotImplementedError

  prediction_prep_fn = functools.partial(ensemble_prep_fn, # add noise
      task_embedder=LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        task_dim=config.word_dim,
        initializer=config.word_initializer,
        compress=config.word_compress),
    )

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vision,
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=prediction_prep_fn,
    prediction=QEnsembleHead(
      num_actions=num_actions,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      policy_layers=config.policy_layers,
      q_input_fn=ConcatFlatStatePolicy(config.state_hidden_size)
      )
  )




def r2d1_gated(config, env_spec):
  num_actions = env_spec.actions.num_values
  task_embedder = LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        task_dim=config.word_dim,
        initializer=config.word_initializer,
        compress=config.word_compress)

  if config.vision_torso == 'atari':
    vision = AtariVisionTorso(
      flatten=False,
      conv_dim=16,
      out_dim=config.vision_size)
  elif config.vision_torso == 'babyai':
    vision = BabyAIVisionTorso(
      flatten=False,
      conv_dim=16,
      batch_norm=config.vision_batch_norm)
  else:
    raise NotImplementedError


  embedder = OAREmbedding(
    num_actions=num_actions,
    concat=False,
    observation=False)
  def gated_attn_prep_fn(inputs, obs):
    task = inputs.observation.task.astype(jnp.uint8)
    has_time = len(task.shape) == 3
    batchfn = hk.BatchApply if has_time else lambda x:x

    def apply_attn(obs, task):
      """Summary
      
      Args:
          obs (TYPE): B x H x W x C
          task (TYPE): B x D
      
      Returns:
          TYPE: Description
      """
      task_embed = task_embedder(task)
      B, D = task_embed.shape
      obs = GatedAttention()(task=task_embed, image=obs)

      obs_flat = obs.reshape(B, -1)

      return obs_flat, task_embed

    action_reward = embedder(inputs, obs)
    obs_flat, task_embed =  batchfn(apply_attn)(obs, task)

    everything = action_reward+[obs_flat, task_embed]
    everything = jnp.concatenate(everything, axis=-1)

    return everything

  pred_prep_fn=None # just use memory output
  
  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vision,
    memory_prep_fn=gated_attn_prep_fn,
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=pred_prep_fn,
    prediction=DuellingMLP(num_actions,
      hidden_sizes=[config.out_hidden_size])
  )
