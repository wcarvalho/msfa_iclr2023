
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

def make_farm_prep_fn(num_actions, task_input=False, embed_task=None):
  """
  Return farm inputs, (1) obs features (2) [action, reward] vector
  """
  embedder = OAREmbedding(
    num_actions=num_actions,
    observation=False,
    concat=False)

  def prep(inputs, obs):
    vector = embedder(inputs)
    if task_input:
      if embed_task is not None:
        raise NotImplementedError("have not checked yet")
      else:
        embed_task = lambda x:x
      task = embed_task(inputs.observation.task)
      vector.append(task)
    vector = jnp.concatenate(vector, axis=-1)
    return farm.FarmInputs(
      image=obs, vector=vector)

  return prep

def flatten_structured_memory(memory_out, **kwargs):
  return memory_out.reshape(*memory_out.shape[:-2], -1)

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
def build_vision_net(config, **kwargs):
  if config.vision_torso == 'atari':
    vision = AtariVisionTorso(**kwargs)
  elif config.vision_torso == 'babyai':
    vision = BabyAIVisionTorso(**kwargs)
  else:
    raise NotImplementedError
  return vision

def build_farm(config, **kwargs):
  return farm.FarmSharedOutput(
    module_size=config.module_size,
    nmodules=config.nmodules,
    image_attn=config.image_attn,
    module_attn_heads=config.module_attn_heads,
    module_attn_size=config.module_attn_size,
    shared_module_attn=config.shared_module_attn,
    projection_dim=config.projection_dim,
    vmap=config.farm_vmap,
    out_layers=config.out_layers,
    **kwargs)

def build_task_embedder(task_embedding, config, lang_task_dim=None, task_dim=None):
  if task_embedding == 'none':
    embedder = embed_fn = Identity(task_dim)
    return embedder, embed_fn
  elif task_embedding == 'language':
    lang_kwargs=dict()
    assert config.task_gate in ['none', 'round', 'sample', 'sigmoid']
    if config.task_gate == 'none':
      pass
    else:
      lang_kwargs['gate_type']=config.task_gate
      lang_kwargs['gates'] = config.nmodules

    embedder = LanguageTaskEmbedder(
        vocab_size=config.max_vocab_size,
        word_dim=config.word_dim,
        sentence_dim=config.word_dim,
        task_dim=lang_task_dim,
        initializer=config.word_initializer,
        compress=config.word_compress,
        tanh=config.lang_tanh,
        **lang_kwargs)
    def embed_fn(task):
      """Convert task to ints, batchapply if necessary, and run through embedding function."""
      has_time = len(task.shape) == 3
      batchfn = hk.BatchApply if has_time else lambda x:x
      return batchfn(embedder)(task.astype(jnp.int32))

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

def r2d1(config, env_spec,
  task_input=True,
  task_embedding: str='none',
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

  # config.lang_task_dim = 0 # use GRU output
  num_actions = env_spec.actions.num_values
  task_dim = env_spec.observations.observation.task.shape[0]
  task_embedder, embed_fn = build_task_embedder(task_embedding, config, lang_task_dim=config.lang_task_dim, task_dim=task_dim)
  inputs_prep_fn = make__convert_floats_embed_task(embed_fn)

  if task_input:
    prediction_prep_fn = r2d1_prediction_prep_fn
  else:
    prediction_prep_fn = None # just use memory_out

  net = BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    **kwargs
  )
  return net

def r2d1_noise(config, env_spec,
  eval_noise=True,
  task_embedding: str='none',
  **net_kwargs):

  # config.lang_task_dim = 0 # use GRU output
  num_actions = env_spec.actions.num_values
  task_dim = env_spec.observations.observation.task.shape[0]
  task_embedder, embed_fn = build_task_embedder(task_embedding, config, lang_task_dim=config.lang_task_dim, task_dim=task_dim)
  inputs_prep_fn = make__convert_floats_embed_task(embed_fn)

  def add_noise_concat(inputs, memory_out, **kwargs):
    """
    Concat [task + noise] with memory output.
    """
    task = inputs.observation.task
    noise = jax.random.normal(hk.next_rng_key(), task.shape)
    task =  task + jnp.sqrt(config.variance) * noise
    return jnp.concatenate((memory_out, task), axis=-1)

  if eval_noise:
    evaluation_prep_fn = add_noise_concat # add noise
  else:
    # seperate eval network that doesn't use noise
    evaluation_prep_fn = r2d1_prediction_prep_fn # don't add noise

  return BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=add_noise_concat, # add noise
    evaluation_prep_fn=evaluation_prep_fn, # (maybe) don't add noise
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    **net_kwargs
  )

def r2d1_farm(config, env_spec,
  task_embedding: str='none',
  **net_kwargs):

  # config.lang_task_dim = 0 # use GRU output
  num_actions = env_spec.actions.num_values
  task_dim = env_spec.observations.observation.task.shape[0]
  task_embedder, embed_fn = build_task_embedder(task_embedding, config, lang_task_dim=config.lang_task_dim, task_dim=task_dim)
  inputs_prep_fn = make__convert_floats_embed_task(embed_fn)

  def r2d1_farm_prediction_prep_fn(inputs, memory_out, **kwargs):
    farm_memory_out = flatten_structured_memory(memory_out)
    return r2d1_prediction_prep_fn(
      inputs=inputs, memory_out=farm_memory_out)

  if config.farm_policy_task_input:
    prediction_prep_fn = r2d1_farm_prediction_prep_fn
  else:
    # only give hidden-state as input
    prediction_prep_fn = flatten_structured_memory

  return BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False, conv_dim=0),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=build_farm(config),
    prediction_prep_fn=prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    **net_kwargs
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
  return USFAInputs(
    w=inputs.observation.task,
    w_train=inputs.observation.train_tasks,
    memory_out=memory_out,
    )

def replace_all_tasks_with_embedding(inputs, embed_fn):
  """Replace both tasks and train tasks."""
  new_observation = inputs.observation._replace(
    task=embed_fn(inputs.observation.task),
    train_tasks=jax.vmap(embed_fn)(inputs.observation.train_tasks))
  return inputs._replace(observation=new_observation)

def usfa(config, env_spec,
  task_embedding='none',
  use_separate_eval=True,
  predict_cumulants=False,
  **net_kwargs):

  num_actions = env_spec.actions.num_values
  vision_net = build_vision_net(config, flatten=True)

  # -----------------------
  # task embedder + prep functions (will embed task)
  # -----------------------
  task_dim = env_spec.observations.observation.task.shape[0]
  task_embedder, embed_fn = build_task_embedder(task_embedding, config, lang_task_dim=config.lang_task_dim, task_dim=task_dim)
  inputs_prep_fn = make__convert_floats_embed_task(embed_fn,
    replace_fn=replace_all_tasks_with_embedding)
  if task_embedding == "language":
    sf_out_dim = task_embedder.out_dim
  elif task_embedding == "none":
    sf_out_dim = task_dim


  prediction_head=UsfaHead(
      num_actions=num_actions,
      state_dim=sf_out_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      z_as_train_task=config.z_as_train_task,
      sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
      multihead=config.multihead,
      concat_w=config.concat_w,
      layernorm=config.sf_layernorm,
      normalize_task=config.normalize_task and config.embed_task,
      )

  if use_separate_eval:
    evaluation_prep_fn=usfa_eval_prep_fn
    evaluation=prediction_head.evaluation
  else:
    evaluation_prep_fn=None
    evaluation=None

  aux_tasks = []
  if predict_cumulants:
    aux_tasks.append(
      CumulantsFromMemoryAuxTask(
        [config.cumulant_hidden_size]*config.cumulant_layers + [sf_out_dim],
        normalize=config.normalize_cumulants,
        activation=config.cumulant_act,
        construction=config.cumulant_const))

  net = BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=get_image_from_inputs,
    vision=vision_net,
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=usfa_prep_fn,
    prediction=prediction_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=evaluation,
    aux_tasks=aux_tasks,
    **net_kwargs
  )
  return net

# ======================================================
# Modular USFA
# ======================================================

def msf(
  config,
  env_spec,
  predict_cumulants=True,
  learn_model=True,
  task_embedding: str='none',
  use_separate_eval=True,
  **net_kwargs):

  assert config.sf_net in ['flat', 'independent', 'relational']
  assert config.phi_net in ['flat', 'independent', 'relational']
  num_actions = env_spec.actions.num_values

  # -----------------------
  # memory
  # -----------------------
  farm_memory = build_farm(config)

  # -----------------------
  # task related
  # -----------------------
  if config.module_task_dim > 0:
    lang_task_dim=config.module_task_dim*config.nmodules
  else:
    lang_task_dim=config.lang_task_dim
  task_dim = env_spec.observations.observation.task.shape[0]
  task_embedder, embed_fn = build_task_embedder(task_embedding, config,
    lang_task_dim=lang_task_dim,
    task_dim=task_dim)

  inputs_prep_fn = make__convert_floats_embed_task(embed_fn,
    replace_fn=replace_all_tasks_with_embedding)


  if task_embedding == "language":
    sf_out_dim = task_embedder.out_dim
  elif task_embedding == "none":
    sf_out_dim = task_dim
    raise RuntimeError("check this")

  # -----------------------
  # USFA Head
  # -----------------------
  usfa_head, pred_prep_fn, eval_prep_fn = build_msf_head(config, sf_out_dim, num_actions)

  if use_separate_eval:
    evaluation_prep_fn=eval_prep_fn
    evaluation=usfa_head.evaluation
  else:
    evaluation_prep_fn=None
    evaluation=None

  # -----------------------
  # Phi head
  # -----------------------
  aux_tasks = []
  if predict_cumulants:
    # takes structured farm input
    phi_net = build_msf_phi_net(config, sf_out_dim)
    aux_tasks.append(phi_net)

  add_bias = getattr(config, "step_penalty", 0) > 0
  if add_bias:
    aux_tasks.append(QBias())

  # -----------------------
  # Model
  # -----------------------
  time_contrast_model = getattr(config, "contrast_time_coeff", 0) > 0
  module_contrast_model = getattr(config, "contrast_module_coeff", 0) > 0
  learn_model = learn_model and (time_contrast_model or module_contrast_model)
  if learn_model:
    # takes structured farm input
    aux_tasks.append(
      FarmModel(
        output_sizes=max(config.model_layers-1, 1)*[config.module_size],
        num_actions=num_actions,
        seperate_params=config.seperate_model_params,
        ),
      )

  return BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=farm_memory,
    prediction_prep_fn=pred_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=evaluation,
    aux_tasks=aux_tasks,
    **net_kwargs
  )

def build_msf_head(config, sf_out_dim, num_actions):

  if config.sf_net == "flat":
    head = UsfaHead(
        num_actions=num_actions,
        state_dim=sf_out_dim,
        hidden_size=config.out_hidden_size,
        policy_size=config.policy_size,
        variance=config.variance,
        nsamples=config.npolicies,
        duelling=config.duelling,
        policy_layers=config.policy_layers,
        z_as_train_task=config.z_as_train_task,
        sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
        multihead=False,
        concat_w=False,
        task_embed=0,
        normalize_task=False,
        layernorm=config.sf_layernorm,
        )
    def pred_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_prep_fn(inputs=inputs, 
        memory_out=flatten_structured_memory(memory_out))

    def eval_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_eval_prep_fn(inputs=inputs, 
        memory_out=flatten_structured_memory(memory_out))

  else:
    if config.sf_net == "independent":
      relational_net = lambda x: x
    elif config.sf_net == "relational":
      relational_net = RelationalNet(
        layers=config.sf_net_layers,
        num_heads=config.sf_net_heads,
        attn_size=config.sf_net_attn_size,
        layernorm=config.layernorm_rel,
        pos_mlp=config.resid_mlp,
        residual=config.relate_residual,
        position_embed=True,
        w_init_scale=config.relate_w_init,
        res_w_init_scale=config.resid_w_init,
        init_bias=config.relate_b_init,
        relu_gate=config.res_relu_gate,
        shared_parameters=not config.seperate_value_params)
    else:
      raise NotImplementedError(config.sf_net)

    hidden_size = config.out_hidden_size if config.out_hidden_size else config.module_size
    head = FarmUsfaHead(
          num_actions=num_actions,
          cumulants_per_module=sf_out_dim//config.nmodules,
          hidden_size=hidden_size,
          policy_size=config.policy_size,
          variance=config.variance,
          nsamples=config.npolicies,
          relational_net=relational_net,
          policy_layers=config.policy_layers,
          multihead=config.seperate_value_params, # seperate params per cumulants
          vmap_multihead=config.farm_vmap,
          layernorm=config.sf_layernorm,
          )
    def pred_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_prep_fn(inputs=inputs, memory_out=memory_out)

    def eval_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_eval_prep_fn(inputs=inputs, memory_out=memory_out)

  return head, pred_prep_fn, eval_prep_fn

def build_msf_phi_net(config, sf_out_dim):
  contrast_module = getattr(config, "contrast_module_coeff", 0) > 0
  contrast_module_delta = getattr(config, "contrast_module_pred", 'delta') == 'delta'
  contrast_module_state = getattr(config, "contrast_module_pred", 'delta') == 'state'
  contrast_time = getattr(config, "contrast_time_coeff", 0) > 0

  normalize_delta = config.normalize_delta and contrast_module and contrast_module_delta
  normalize_state = config.normalize_state and (contrast_module_state or contrast_time)

  if config.phi_net == "flat":
    return FarmCumulants(
          activation=config.cumulant_act,
          module_cumulants=sf_out_dim,
          hidden_size=config.cumulant_hidden_size,
          layers=config.cumulant_layers,
          aggregation='concat',
          normalize_cumulants=config.normalize_cumulants,
          normalize_delta=normalize_delta,
          construction=config.cumulant_const,
          )
  else:
    if config.phi_net == "independent":
      relational_net = lambda x: x
    elif config.phi_net == "relational":
      relational_net = RelationalNet(
        layers=config.phi_net_layers,
        num_heads=config.phi_net_heads,
        residual=config.relate_residual,
        layernorm=config.layernorm_rel,
        pos_mlp=config.resid_mlp,
        w_init_scale=config.relate_w_init,
        res_w_init_scale=config.resid_w_init,
        init_bias=config.relate_b_init,
        relu_gate=config.res_relu_gate,
        shared_parameters=not config.seperate_cumulant_params)
    else:
      raise NotImplementedError(config.phi_net)

    return FarmIndependentCumulants(
        activation=config.cumulant_act,
        module_cumulants=sf_out_dim//config.nmodules,
        hidden_size=config.cumulant_hidden_size,
        layers=config.cumulant_layers,
        seperate_params=config.seperate_cumulant_params,
        construction=config.cumulant_const,
        relational_net=relational_net,
        normalize_delta=normalize_delta,
        normalize_state=normalize_state,
        normalize_cumulants=config.normalize_cumulants)
  raise RuntimeError
