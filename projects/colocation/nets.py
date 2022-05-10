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
from modules.usfa import UsfaHead, USFAInputs, ConcatFlatStatePolicy, CumulantsFromMemoryAuxTask
from projects.colocation.cumulants import CumulantsFromConvTask, LinearTaskEmbed
from acme.adders import reverb as adders_reverb

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

def r2d1_prediction_prep_fn(inputs, memory_out, **kwargs):
  """
  Concat task with memory output.
  """
  task = inputs.observation.task
  return jnp.concatenate((memory_out, task), axis=-1)

# ======================================================
# Networks
# ======================================================

def r2d1(config, env_spec, **kwargs):
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

def usfa(config, env_spec, use_seperate_eval=True, predict_cumulants = False,**kwargs):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]
  task_embed = 0
  if predict_cumulants:
    task_embed = LinearTaskEmbed(config.cumulant_dimension)
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
      task_embed=task_embed,
    ##SPECIAL TIME
      train_task_as_z=predict_cumulants
      )
  aux_tasks = []
  if predict_cumulants:
    # aux_tasks.append(
    #   CumulantsFromMemoryAuxTask(
    #     [config.cumulant_hidden_size, state_dim],
    #     normalize=config.normalize_cumulants,
    #     activation=config.cumulant_act,
    #     construction=config.cumulant_const))
    aux_tasks.append(
      CumulantsFromConvTask(
        [config.cumulant_hidden_size, config.cumulant_dimension],
        normalize=config.normalize_cumulants,
        activation=config.cumulant_act))
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

def r2d1_noise(config, env_spec, **kwargs):
  num_actions = env_spec.actions.num_values

  def add_noise_concat(inputs, memory_out, **kwargs):
    """
    Concat [task + noise] with memory output.
    """
    task = inputs.observation.task
    noise = jax.random.normal(hk.next_rng_key(), task.shape)
    task =  task + jnp.sqrt(config.variance) * noise
    return jnp.concatenate((memory_out, task), axis=-1)

  if config.eval_network:
    # seperate eval network that doesn't use noise
    evaluation_prep_fn = r2d1_prediction_prep_fn # don't add noise
  else:
    evaluation_prep_fn = add_noise_concat # add noise

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=add_noise_concat, # add noise
    evaluation_prep_fn=evaluation_prep_fn, # (maybe) don't add noise
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )