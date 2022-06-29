
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
from modules.embedding import BabyAIEmbedding, OAREmbedding, LanguageTaskEmbedder, Identity, LinearTaskEmbedding
from modules import farm
from modules.relational import RelationalLayer, RelationalNet
from modules.farm_model import FarmModel, FarmCumulants, FarmIndependentCumulants
from modules.farm_usfa import FarmUsfaHead, RelationalFarmUsfaHead
from modules.farm_uvfa import FarmUvfaHead, FarmUvfaInputs

from modules.vision import AtariVisionTorso, BabyAIVisionTorso, BabyAIymbolicVisionTorso
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

def get_obs_from_inputs(inputs : observation_action_reward.OAR):
  return inputs.observation

def make_farm_prep_fn(num_actions, task_input=False, embed_task=None, symbolic=False):
  """
  Return farm inputs, (1) obs features (2) [action, reward] vector
  """
  embedder = BabyAIEmbedding(
    num_actions=num_actions,
    observation=False,
    concat=False,
    symbolic=symbolic)

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
def build_vision_net(config, env_spec, **kwargs):
  if getattr(config, "symbolic", False):
    config.vision_torso = 'symbolic'
    num_symbols = env_spec.observations.observation.image.maximum.max()
    num_channels = len(env_spec.observations.observation.image.shape)
    vision = BabyAIymbolicVisionTorso(
      # num_channels=num_channels,
      num_symbols=num_symbols, **kwargs)
  else:
    if config.vision_torso == 'atari':
      vision = AtariVisionTorso(**kwargs)
    elif config.vision_torso == 'babyai':
      vision = BabyAIVisionTorso(**kwargs)
    else:
      raise NotImplementedError
  return vision

def build_vision_prep_fn(config):
  if getattr(config, "symbolic", False):
    return get_obs_from_inputs
  else:
    return get_image_from_inputs

def build_farm(config, **kwargs):
  farm_memory = farm.FarmSharedOutput(
    module_size=config.module_size,
    nmodules=config.nmodules,
    memory_size=config.memory_size,
    image_attn=config.image_attn,
    module_attn_heads=config.module_attn_heads,
    module_attn_size=config.module_attn_size,
    shared_module_attn=config.shared_module_attn,
    projection_dim=config.projection_dim,
    vmap=config.farm_vmap,
    out_layers=config.out_layers,
    **kwargs)

  config.nmodules = farm_memory.nmodules
  config.memory_size = farm_memory.memory_size
  config.module_size = farm_memory.module_size

  # -----------------------
  # task related
  # -----------------------
  if getattr(config, 'module_task_dim', 0) > 0:
    config.embed_task_dim=config.module_task_dim*config.nmodules

  return farm_memory


def build_task_embedder(task_embedding, config, task_shape=None):
  num_tasks = task_shape[-1]
  if task_embedding == 'none':
    assert len(task_shape) == 1, "don't know how to handle"
    embedder = embed_fn = Identity(num_tasks)
    return embedder, embed_fn
  elif task_embedding in ['embedding', 'struct_embed']:
    structured = task_embedding == 'struct_embed'
    embedder = LinearTaskEmbedding(
      num_tasks=num_tasks,
      hidden_dim=config.word_dim,
      out_dim=config.embed_task_dim,
      structured=structured)
    assert len(task_shape) == 1, "don't know how to handle"
    def embed_fn(task):
      """Convert task to ints, batchapply if necessary, and run through embedding function."""
      has_time = len(task.shape) == 3
      batchfn = hk.BatchApply if has_time else lambda x:x
      return batchfn(embedder)(task)
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
        bow=config.bag_of_words,
        task_dim=config.embed_task_dim,
        initializer=config.word_initializer,
        compress=config.word_compress,
        activation=config.lang_activation,
        **lang_kwargs)
    def embed_fn(task):
      """Convert task to ints, batchapply if necessary, and run through embedding function."""
      if len(task.shape) == (len(task_shape) + 2):
        # has (T, B) 
        has_time = True
      elif len(task.shape) == (len(task_shape) + 1):
        # has (B)
        has_time = False
      elif len(task.shape) == (len(task_shape)):
        # has (B)
        has_time = False
      else:
        raise RuntimeError
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
  task_input: str='qfn',
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

  # config.embed_task_dim = 0 # use GRU output
  num_actions = env_spec.actions.num_values
  task_shape = env_spec.observations.observation.task.shape
  task_dim = task_shape[-1]

  inputs_prep_fn = convert_floats
  if task_input != 'none':
    task_embedder, embed_fn = build_task_embedder(
      task_embedding=task_embedding,
      config=config,
      task_shape=task_shape)
    if task_embedding != 'none':
      inputs_prep_fn = make__convert_floats_embed_task(embed_fn)

    embedder = BabyAIEmbedding(num_actions=num_actions)
    def task_in_memory_prep_fn(inputs, obs):
      task = inputs.observation.task
      oar = embedder(inputs, obs)
      return jnp.concatenate((oar, task), axis=-1)

  if task_input == 'qfn':
    memory_prep_fn = embedder
    prediction_prep_fn = r2d1_prediction_prep_fn
  elif task_input == 'memory':
    memory_prep_fn=task_in_memory_prep_fn
    prediction_prep_fn = None # just use memory_out
  elif task_input == 'none':
    prediction_prep_fn = None # just use memory_out
    memory_prep_fn=None
  else:
    raise RuntimeError(task_input)

  net = BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=build_vision_prep_fn(config),
    vision=build_vision_net(config, env_spec, flatten=True),
    memory_prep_fn=memory_prep_fn,
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

  # config.embed_task_dim = 0 # use GRU output
  num_actions = env_spec.actions.num_values
  task_shape = env_spec.observations.observation.task.shape
  task_dim = task_shape[-1]
  task_embedder, embed_fn = build_task_embedder(task_embedding=task_embedding, config=config, task_shape=task_shape)
  if task_embedding != 'none':
    inputs_prep_fn = make__convert_floats_embed_task(embed_fn)
  else:
    inputs_prep_fn = convert_floats

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
    vision_prep_fn=build_vision_prep_fn(config),
    vision=build_vision_net(config, env_spec, flatten=True),
    memory_prep_fn=BabyAIEmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=add_noise_concat, # add noise
    evaluation_prep_fn=evaluation_prep_fn, # (maybe) don't add noise
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    **net_kwargs
  )

# ======================================================
# Modular R2D1
# ======================================================

def modr2d1(config, env_spec,
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
  farm_memory = build_farm(config, return_attn=True)


  num_actions = env_spec.actions.num_values
  task_shape = env_spec.observations.observation.task.shape
  task_dim = task_shape[-1]
  task_embedder, embed_fn = build_task_embedder(
    task_embedding=task_embedding,
    config=config,
    task_shape=task_shape)
  if task_embedding != 'none':
    inputs_prep_fn = make__convert_floats_embed_task(embed_fn)
  else:
    inputs_prep_fn = convert_floats

  def prediction_prep_fn(inputs, memory_out, **kwargs):
    """
    Concat [task + noise] with memory output.
    """
    return FarmUvfaInputs(
      w=inputs.observation.task,
      memory_out=memory_out,
      )

  net = BasicRecurrent(
    inputs_prep_fn=inputs_prep_fn,
    vision_prep_fn=build_vision_prep_fn(config),
    vision=build_vision_net(config, env_spec, flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input,
    ),
    memory=farm_memory,
    prediction_prep_fn=prediction_prep_fn,
    prediction=FarmUvfaHead(
      num_actions=num_actions,
      hidden_sizes=[config.out_hidden_size],
      task_embed_dim=config.policy_size,
      task_embed_layers=config.policy_layers,
      struct_w_input=config.struct_w,
      dot_heads=config.dot_qheads),
    **kwargs
  )
  return net

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
  vision_net = build_vision_net(config, env_spec, flatten=True)

  # -----------------------
  # task embedder + prep functions (will embed task)
  # -----------------------
  task_shape = env_spec.observations.observation.task.shape
  task_dim = task_shape[-1]
  task_embedder, embed_fn = build_task_embedder(
    task_embedding=task_embedding,
    config=config,
    task_shape=task_shape)
  if task_embedding != 'none':
    inputs_prep_fn = make__convert_floats_embed_task(embed_fn, replace_fn=replace_all_tasks_with_embedding)
  else:
    inputs_prep_fn = convert_floats

  if task_embedding == "none":
    sf_out_dim = task_dim
  else:
    sf_out_dim = task_embedder.out_dim


  prediction_head=UsfaHead(
      num_actions=num_actions,
      state_dim=sf_out_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      stop_z_grad=config.stop_z_grad,
      z_as_train_task=config.z_as_train_task,
      sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
      multihead=config.multihead,
      concat_w=config.concat_w,
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
    vision_prep_fn=build_vision_prep_fn(config),
    vision=vision_net,
    memory_prep_fn=BabyAIEmbedding(
      num_actions=num_actions,
    ),
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

  assert config.sf_net in ['flat', 'independent', 'relational', 'relational_action']
  assert config.phi_net in ['flat', 'independent', 'relational']
  num_actions = env_spec.actions.num_values
  task_shape = env_spec.observations.observation.task.shape
  task_dim = task_shape[-1]


  # -----------------------
  # memory
  # -----------------------
  # ensure sizes are correct
  if task_embedding == 'none' and task_dim < config.nmodules:
    # if not embedding and don't have enough modules, reduce
    module_size = config.module_size
    if config.module_size is None:
      module_size = config.memory_size//config.nmodules
    config.nmodules = task_dim
    config.memory_size = config.nmodules*module_size


  farm_memory = build_farm(config, return_attn=True)

  task_embedder, embed_fn = build_task_embedder(
    task_embedding=task_embedding,
    config=config,
    task_shape=task_shape)

  if task_embedding != 'none':
    inputs_prep_fn = make__convert_floats_embed_task(embed_fn,
    replace_fn=replace_all_tasks_with_embedding)
  else:
    inputs_prep_fn = convert_floats

  if task_embedding == "none":
    sf_out_dim = task_dim
  else:
    sf_out_dim = task_embedder.out_dim

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
    vision_prep_fn=build_vision_prep_fn(config),
    vision=build_vision_net(config, env_spec, flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input,
    ),
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
        stop_z_grad=config.stop_z_grad,
        z_as_train_task=config.z_as_train_task,
        sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
        multihead=False,
        concat_w=False,
        task_embed=0,
        normalize_task=False,
        )
    def pred_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_prep_fn(inputs=inputs, 
        memory_out=flatten_structured_memory(memory_out.hidden))

    def eval_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_eval_prep_fn(inputs=inputs, 
        memory_out=flatten_structured_memory(memory_out.hidden))

  else:
    if config.sf_net == "relational_action":
      hidden_size = config.out_hidden_size if config.out_hidden_size else config.module_size
      config.sf_net_heads = config.nmodules//2
      config.sf_net_attn_size = config.module_size*config.sf_net_heads

      relational_net = RelationalLayer(
          num_heads=config.sf_net_heads,
          attn_size=config.sf_net_attn_size,
          layernorm=config.layernorm_rel,
          pos_mlp=config.resid_mlp,
          residual=config.relate_residual,
          position_embed=config.relation_position_embed,
          w_init_scale=config.relate_w_init,
          res_w_init_scale=config.resid_w_init,
          init_bias=config.relate_b_init,
          relu_gate=config.res_relu_gate,
          shared_parameters=True)
      head = RelationalFarmUsfaHead(
            num_actions=num_actions,
            cumulants_per_module=sf_out_dim//config.nmodules,
            hidden_size=hidden_size,
            policy_size=config.policy_size,
            variance=config.variance,
            nsamples=config.npolicies,
            relational_net=relational_net,
            policy_layers=config.policy_layers,
            stop_z_grad=config.stop_z_grad,
            struct_policy=config.struct_policy_input,
            eval_task_support=config.eval_task_support,
            multihead=config.seperate_value_params, # seperate params per cumulants
            # vmap_multihead=config.farm_vmap,
            # position_embed=config.embed_position,
            )
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
          position_embed=config.relation_position_embed,
          w_init_scale=config.relate_w_init,
          res_w_init_scale=config.resid_w_init,
          init_bias=config.relate_b_init,
          relu_gate=config.res_relu_gate,
          shared_parameters=True)
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
            stop_z_grad=config.stop_z_grad,
            struct_policy=config.struct_policy_input,
            eval_task_support=config.eval_task_support,
            multihead=config.seperate_value_params, # seperate params per cumulants
            # vmap_multihead=config.farm_vmap,
            # position_embed=config.embed_position,
            )
    def pred_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_prep_fn(
        inputs=inputs,
        memory_out=memory_out.hidden)

    def eval_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_eval_prep_fn(
        inputs=inputs,
        memory_out=memory_out.hidden)

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
        shared_parameters=True)
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
