
from typing import Callable, Optional, Tuple, NamedTuple

from acme.jax.networks import duelling
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import functools

from agents import td_agent
from agents.td_agent.types import Predictions
from modules.basic_archs import BasicRecurrent
from modules.embedding import OAREmbedding, LanguageTaskEmbedder, Identity
from modules import farm
from modules.relational import RelationalLayer, RelationalNet
from modules.farm_model import FarmModel, FarmCumulants, FarmIndependentCumulants
from modules.farm_usfa import FarmUsfaHead

from modules.vision import AtariVisionTorso, BabyAIVisionTorso
from modules.usfa import UsfaHead, USFAInputs, CumulantsFromMemoryAuxTask, ConcatFlatStatePolicy, UniqueStatePolicyPairs, QBias
from modules.ensembles import QEnsembleInputs, QEnsembleHead
from modules import usfa as usfa_modules
from modules import vae as vae_modules

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

def replace_task_with_embedding(inputs, embed_fn):
    new_observation = inputs.observation._replace(
      task=embed_fn(inputs.observation.task))
    return inputs._replace(observation=new_observation)

def make__convert_floats_embed_task(embed_fn,
  replace_fn=replace_task_with_embedding):
  def inputs_prep_fn(inputs):
    inputs = convert_floats(inputs)
    inputs = replace_fn(inputs, embed_fn)
    return inputs

  return inputs_prep_fn

# ======================================================
# Building Modules
# ======================================================


def build_task_embedder(task_embedding, config, lang_task_dim=None, task_dim=None):
  if task_embedding == 'none':
    embedder = embed_fn = Identity(task_dim)
    return embedder, embed_fn
  elif task_embedding == 'language':
    embedder = LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        sentence_dim=config.word_dim,
        task_dim=lang_task_dim,
        initializer=config.word_initializer,
        compress=config.word_compress,
        tanh=config.lang_tanh,
        relu=config.lang_relu)
    def embed_fn(task):
      """Convert task to ints, batchapply if necessary, and run through embedding function."""
      has_time = len(task.shape) == 3
      batchfn = hk.BatchApply if has_time else lambda x:x
      new_embedder = batchfn(embedder)
      return new_embedder(task.astype(jnp.int32))

    return embedder, embed_fn

  else:
    raise NotImplementedError(task_embedding)



# ======================================================
# R2D1
# ======================================================

def r2d1_prediction_prep_fn(inputs, memory_out, **kwargs):
  """
  Concat task with memory output.
  """
  task = inputs.observation.task
  return jnp.concatenate((memory_out, task), axis=-1)

def r2d1(config, env_spec, task_embedding: str='language',
   **kwargs):
  """Summary
  
  Args:
      config (TYPE): Description
      env_spec (TYPE): Description
      task_input (bool, optional): whether to give task as input to agent. No=basic baseline.
      task_embedding (str, optional): Options: [identity, language]
      **kwargs: Description
  
  Returns:
      TYPE: Description
  """

  num_actions = env_spec.actions.num_values
  task_dim = env_spec.observations.observation.task.shape[0]
  task_embedder, embed_fn = build_task_embedder(
    task_embedding=task_embedding,
    config=config,
    lang_task_dim=config.lang_task_dim,
    task_dim=task_dim)
  inputs_prep_fn = make__convert_floats_embed_task(embed_fn)

  memory_prep_fn = OAREmbedding(num_actions=num_actions)
  prediction_prep_fn = r2d1_prediction_prep_fn

  net = BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=memory_prep_fn,
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    **kwargs
  )
  return net
