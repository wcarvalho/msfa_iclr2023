import haiku as hk

from typing import Callable, Optional, Tuple, NamedTuple

from acme.jax.networks import duelling
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp

from agents.td_agent.types import Predictions
from modules.basic_archs import BasicRecurrent
from modules.embedding import OAREmbedding
from modules.farm import FARM, FarmInputs
from modules.vision import AtariVisionTorso
from modules.usfa import UsfaHead, USFAInputs
from utils import data as data_utils

class DuellingMLP(duelling.DuellingMLP):
  def __call__(self, *args, **kwargs):
    kwargs.pop("key", None)
    q = super().__call__(*args, **kwargs)
    return Predictions(q=q)

# ======================================================
# Processing functions
# ======================================================

def make_floats(inputs):
  return jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

def get_image_from_inputs(inputs : observation_action_reward.OAR):
  return inputs.observation.image/255.0

def farm_prep_fn(inputs, obs):
  raise RuntimeError("state features if R2D1, not if USFA")
  return FarmInputs(
    image=obs, vector=oar_flatten(inputs, obs, observation=False))

def flatten_structured_memory(memory_out, **kwargs):
  return memory_out.reshape(*memory_out.shape[:-2], -1)


# ======================================================
# R2D1
# ======================================================

def make_r2d1_lstm_prep_fn(num_actions):
  """
  Creat function to concat [obs, action, reward] from `oar_prep` with state features (phi).
  """
  embedder = OAREmbedding(num_actions=num_actions, concat=False)
  def prep(inputs, obs):
    items = [inputs.observation.state_features]
    items.extend(embedder(inputs, obs))

    return jnp.concatenate(items, axis=-1)

  return prep

def r2d1_prediction_prep_fn(inputs, memory_out, **kwargs):
  """
  Concat task with memory output.
  """
  task = inputs.observation.task
  return jnp.concatenate((memory_out, task), axis=-1)

def make_r2d1(config, env_spec):
  num_actions = env_spec.actions.num_values

  return BasicRecurrent(
    inputs_prep_fn=make_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=make_r2d1_lstm_prep_fn(num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=r2d1_prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

# -----------------------
# R2D1 + FARM
# -----------------------
def make_r2d1_farm_prep_fn(num_actions):
  """
  Return farm inputs, (1) obs features (2) [action, reward] vector
  """
  embedder = OAREmbedding(num_actions=num_actions, observation=False)
  def prep(inputs, obs):
    return FarmInputs(
      image=obs, vector=embedder(inputs))

  return prep

def make_r2d1_farm(config, env_spec):
  num_actions = env_spec.actions.num_values

  return BasicRecurrent(
    inputs_prep_fn=make_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_r2d1_farm_prep_fn(num_actions),
    memory=FARM(config.module_size, config.nmodules),
    prediction_prep_fn=flatten_structured_memory,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

# ======================================================
# USFA
# ======================================================
def usfa_prep_fn(inputs, memory_out, *args, **kwargs):
  return USFAInputs(
    w=inputs.observation.task,
    memory_out=memory_out,
    )

def make_usfa(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  return BasicRecurrent(
    inputs_prep_fn=make_floats,
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

