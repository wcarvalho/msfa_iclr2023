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
from modules.usfa import UsfaHead, USFAInputs, RewardAuxTask, ValueAuxTask
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
  embedder = OAREmbedding(num_actions=num_actions, observation=False)
  def prep(inputs, obs):
    return farm.FarmInputs(
      image=obs, vector=embedder(inputs))

  return prep

def flatten_structured_memory(memory_out, **kwargs):
  return memory_out.reshape(*memory_out.shape[:-2], -1)

def memory_prep_fn(num_actions, extract_fn):
  """Combine vae samples w/ action + reward"""
  embedder = OAREmbedding(
    num_actions=num_actions,
    observation=False,
    concat=False)
  def prep(inputs, obs):
    items = [extract_fn(inputs, obs)]
    items.extend(embedder(inputs))

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
    memory_prep_fn=memory_prep_fn(
      num_actions=num_actions,
      extract_fn=lambda inputs, obs: inputs.observation.state_features),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=r2d1_prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

def r2d1_vae(config, env_spec):
  num_actions = env_spec.actions.num_values

  prediction = DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  vae = vae_modules.VAE(
    latent_dim=config.latent_dim,
    latent_source=config.latent_source,
    **vae_modules.small_standard_encoder_decoder(),
    )
  aux_tasks = vae.aux_task

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vae,
    memory_prep_fn=memory_prep_fn(
      num_actions=num_actions,
      extract_fn=lambda inputs, obs: obs.samples),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=r2d1_prediction_prep_fn,
    prediction=prediction,
    aux_tasks=aux_tasks,
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
      num_actions=num_actions),
  )

# ======================================================
# USFA
# ======================================================
def usfa_prep_fn(inputs, memory_out, *args, **kwargs):
  return USFAInputs(
    w=inputs.observation.task,
    memory_out=memory_out,
    )

# def usfa_farm_prediction_prep_fn(inputs, memory_out, *args, **kwargs):
#   return USFAInputs(
#     w=inputs.observation.task,
#     memory_out=memory_out,
#     )

def usfa(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=usfa_prep_fn,
    prediction=UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      )
  )

def usfa_reward_vae(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]
  prediction = UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      )

  vae = VAE(latent_dim=config.latent_dim)

  aux_tasks = [
    vae.aux_task,
    RewardAuxTask([state_dim])
  ]

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vae,
    memory_prep_fn=memory_prep_fn(
      num_actions=num_actions,
      extract_fn=lambda inputs, obs: obs.samples),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=usfa_prep_fn,
    prediction=prediction,
    aux_tasks=aux_tasks,
  )

def usfa_farmflat_model(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  aux_tasks = [
    FarmModel(
      config.model_layers*[config.module_size],
      num_actions=num_actions),
    FarmCumulants([config.out_hidden_size, state_dim], cumtype='sum'),
  ]
  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=farm.FARM(config.module_size, config.nmodules),
    memory_proc_fn=flatten_structured_memory,
    prediction_prep_fn=usfa_prep_fn,
    prediction=UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      ),
    aux_tasks=aux_tasks,
  )
