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
from modules.usfa import UsfaHead, USFAInputs, CumulantsAuxTask, ConcatFlatStatePolicy

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
  #literally just grab the task vector
  task = inputs.observation.task
  return task_embedder(task).astype(jnp.int32)

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
      task_embedder=lambda x: x #stupid simple task embedder lol
      ),
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

def usfa_prep_fn(inputs, memory_out, *args, **kwargs):
  return USFAInputs(
    w=inputs.observation.task,
    memory_out=memory_out,
    )

def usfa_eval_prep_fn(inputs, memory_out, *args, **kwargs):
  dim = inputs.observation.task.shape[-1]
  w_train = jnp.identity(dim)
  return USFAInputs(
    w=inputs.observation.task,
    w_train=w_train,
    memory_out=memory_out,
    )

def usfa(config, env_spec, use_seperate_eval=True):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]
  prediction_head=UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      z_as_train_task=config.z_as_train_task,
      sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
      )

  if use_seperate_eval:
    evaluation_prep_fn=usfa_eval_prep_fn
    evaluation=prediction_head.evaluation
  else:
    evaluation_prep_fn=None
    evaluation=None

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=usfa_prep_fn,
    prediction=prediction_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=evaluation,
  )