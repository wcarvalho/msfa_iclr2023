import haiku as hk

from typing import Callable, Optional, Tuple, NamedTuple

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
from modules.embedding import OAREmbedding
from modules import farm
from modules.farm_model import FarmModel, FarmCumulants
from modules.vision import AtariVisionTorso
from modules.usfa import UsfaHead, USFAInputs, USFAInputs, CumulantsAuxTask, ConcatFlatStatePolicy
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

def make_farm_prep_fn(num_actions):
  """
  Return farm inputs, (1) obs features (2) [action, reward] vector
  """
  embedder = OAREmbedding(
    num_actions=num_actions,
    observation=False)
  def prep(inputs, obs):
    return farm.FarmInputs(
      image=obs, vector=embedder(inputs))

  return prep

def flatten_structured_memory(memory_out, **kwargs):
  return memory_out.reshape(*memory_out.shape[:-2], -1)

def memory_prep_fn(num_actions, extract_fn=None):
  """Combine vae samples w/ action + reward"""
  embedder = OAREmbedding(
    num_actions=num_actions,
    observation=True,
    concat=False)
  def prep(inputs, obs):
    if extract_fn:
      items = [extract_fn(inputs, obs)]
    else:
      items = []
    items.extend(embedder(inputs, obs))

    return jnp.concatenate(items, axis=-1)

  return prep

# ======================================================
# R2D1
# ======================================================


def r2d1_prediction_prep_fn(inputs, memory_out, **kwargs):
  """
  Concat task with memory output.
  """
  task = inputs.observation.task
  return jnp.concatenate((memory_out, task), axis=-1)

def r2d1(config, env_spec):
  num_actions = env_spec.actions.num_values

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=r2d1_prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

def r2d1_farm(config, env_spec):
  num_actions = env_spec.actions.num_values

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=farm.FARM(config.module_size, config.nmodules),
    prediction_prep_fn=flatten_structured_memory,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

def r2d1_farm_model(config, env_spec):
  num_actions = env_spec.actions.num_values


  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=farm.FarmSharedOutput(
      module_size=config.module_size,
      nmodules=config.nmodules,
      out_layers=config.out_layers),
    prediction_prep_fn=flatten_structured_memory,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    aux_tasks=FarmModel(
      config.model_layers*[config.module_size],
      num_actions=num_actions,
      activation=getattr(jax.nn, config.activation)),
  )

# ======================================================
# USFA
# ======================================================
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


def build_usfa_farm_head(config, state_dim, num_actions, farm_memory, flat=True):

  if config.embed_task:
    # if embedding task, don't project delta for cumulant
    # embed task to size of delta
    if flat:
      task_embed = farm_memory.total_dim
    else:
      task_embed = farm_memory.module_size
  else:
    # if not embedding task, project delta for cumulant
    task_embed = 0

  usfa_head = UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      z_as_train_task=config.z_as_train_task,
      task_embed=task_embed,
      normalize_task=config.normalize_task and config.embed_task,
      )
  return usfa_head

def usfa_farmflat_model(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  farm_memory = farm.FarmSharedOutput(
      module_size=config.module_size,
      nmodules=config.nmodules,
      out_layers=config.out_layers)

  usfa_head = build_usfa_farm_head(
    config=config, state_dim=state_dim, num_actions=num_actions,
    farm_memory=farm_memory, flat=True)

  aux_tasks = [
    # takes structured farm input
    FarmModel(
      config.model_layers*[config.module_size],
      num_actions=num_actions,
      activation=getattr(jax.nn, config.activation)),
    # takes structured farm input
    FarmCumulants(
      out_dim=usfa_head.out_dim,
      hidden_size=config.cumulant_hidden_size,
      cumtype='concat',
      normalize_cumulants=config.normalize_cumulants),
  ]

  def prediction_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_prep_fn(inputs=inputs, 
      memory_out=flatten_structured_memory(memory_out))

  def evaluation_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_eval_prep_fn(inputs=inputs, 
      memory_out=flatten_structured_memory(memory_out))

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=farm_memory,
    prediction_prep_fn=prediction_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=usfa_head.evaluation,
    aux_tasks=aux_tasks,
  )
