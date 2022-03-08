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
from modules.vision import AtariVisionTorso
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
  return batchfn(task_embedder)(task.astype(jnp.int32))

def prediction_prep_fn(inputs, memory_out, task_embedder, **kwargs):
  """
  Concat task with memory output.
  """
  task = embed_task(inputs, task_embedder)
  return jnp.concatenate((memory_out, task), axis=-1)

# ======================================================
# Networks
# ======================================================

def r2d1(config, env_spec):
  num_actions = env_spec.actions.num_values

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=functools.partial(prediction_prep_fn,
      task_embedder=LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        task_dim=config.word_dim),
      ),
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

def r2d1_noise(config, env_spec):
  num_actions = env_spec.actions.num_values

  def add_noise_concat(inputs, memory_out, task_embedder, **kwargs):
    """
    1. embed task
    2. Concat [task + noise] with memory output.
    """
    task = embed_task(inputs, task_embedder)
    noise = jax.random.normal(hk.next_rng_key(), task.shape)
    task =  task + jnp.sqrt(config.variance) * noise
    return jnp.concatenate((memory_out, task), axis=-1)

  new_add_noise_concat=functools.partial(add_noise_concat, # add noise
      task_embedder=LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        task_dim=config.word_dim),
    )

  if config.eval_network:
    # seperate eval network that doesn't use noise
    evaluation_prep_fn=functools.partial(prediction_prep_fn, # don't add noise
        task_embedder=LanguageTaskEmbedder(
          vocab_size=config.max_vocab_size,
          word_dim=config.word_dim,
          task_dim=config.word_dim),
      )
  else:
    evaluation_prep_fn = new_add_noise_concat # add noise

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=prediction_prep_fn,
    evaluation_prep_fn=evaluation_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )


def r2d1_noise_ensemble(config, env_spec):
  num_actions = env_spec.actions.num_values

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

  prediction_prep_fn = functools.partial(ensemble_prep_fn, # add noise
      task_embedder=LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        task_dim=config.word_dim),
    )

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
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