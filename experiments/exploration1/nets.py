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
from modules.relational import RelationalLayer, RelationalNet
from modules.farm_model import FarmModel, FarmCumulants, FarmIndependentCumulants
from modules.farm_usfa import FarmUsfaHead

from modules.vision import AtariVisionTorso, BabyAIVisionTorso, AtariImpalaTorso
from modules.usfa import UsfaHead, USFAInputs, CumulantsFromMemoryAuxTask, ConcatFlatStatePolicy, UniqueStatePolicyPairs
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

def make_farm_prep_fn(num_actions, task_input=True):
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
      vector.append(inputs.observation.task)
    vector = jnp.concatenate(vector, axis=-1)
    return farm.FarmInputs(
      image=obs, vector=vector)

  return prep

def flatten_structured_memory(memory_out, **kwargs):
  return memory_out.reshape(*memory_out.shape[:-2], -1)

def build_vision_net(config, **kwargs):
  if config.vision_torso == 'atari':
    vision = AtariVisionTorso(**kwargs)
  elif config.vision_torso == 'babyai':
    vision = BabyAIVisionTorso(**kwargs)
  elif config.vision_torso == 'impala':
    vision = AtariImpalaTorso(**kwargs)
  else:
    raise NotImplementedError
  return vision

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
    share_residual=config.share_residual,
    share_init_bias=config.share_init_bias,
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

# ======================================================
# R2D1
# ======================================================

def r2d1_prediction_prep_fn(inputs, memory_out, **kwargs):
  """
  Concat task with memory output.
  """
  task = inputs.observation.task
  return jnp.concatenate((memory_out, task), axis=-1)

def r2d1(config, env_spec, task_input=True, **kwargs):
  num_actions = env_spec.actions.num_values
  if task_input:
    prediction_prep_fn=r2d1_prediction_prep_fn
  else:
    prediction_prep_fn = None # just use memory_out

  net = BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=build_vision_net(config, flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=prediction_prep_fn,
    prediction=DuellingMLP(num_actions,
        hidden_sizes=[config.out_hidden_size]*config.out_q_layers),
    **kwargs
  )
  return net

def r2d1_noise(config, env_spec, eval_noise=True, **net_kwargs):
  num_actions = env_spec.actions.num_values

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
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=add_noise_concat, # add noise
    evaluation_prep_fn=evaluation_prep_fn, # (maybe) don't add noise
    prediction=DuellingMLP(num_actions,
        hidden_sizes=[config.out_hidden_size]*config.out_q_layers),
    **net_kwargs
  )

def r2d1_farm(config, env_spec, **net_kwargs):
  num_actions = env_spec.actions.num_values

  if config.farm_policy_task_input:
    def prediction_prep_fn(inputs, memory_out, **kwargs):
      farm_memory_out = flatten_structured_memory(memory_out.hidden)
      return r2d1_prediction_prep_fn(
        inputs=inputs, memory_out=farm_memory_out)
  else:
    # only give hidden-state as input
    def prediction_prep_fn(inputs, memory_out, **kwargs):
      return flatten_structured_memory(memory_out.hidden)

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False, conv_dim=0),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=build_farm(config, return_attn=True),
    prediction_prep_fn=prediction_prep_fn,
    prediction=DuellingMLP(num_actions,
        hidden_sizes=[config.out_hidden_size]*config.out_q_layers),
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
  dim = inputs.observation.task.shape[-1]
  w_train = jnp.identity(dim).astype(jnp.float32)
  return USFAInputs(
    w=inputs.observation.task,
    w_train=w_train,
    memory_out=memory_out,
    )


def usfa(config, env_spec, use_seperate_eval=True, predict_cumulants=False, **net_kwargs):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.task.shape[0]

  vision_net = build_vision_net(config, flatten=True)

  prediction_head=UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      head_layers=config.out_q_layers,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      z_as_train_task=config.z_as_train_task,
      sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
      multihead=config.multihead,
      concat_w=config.concat_w,
      eval_task_support=config.eval_task_support,
      normalize_task=config.normalize_task and config.embed_task,
      )

  if use_seperate_eval:
    evaluation_prep_fn=usfa_eval_prep_fn
    evaluation=prediction_head.evaluation
  else:
    evaluation_prep_fn=None
    evaluation=None

  aux_tasks = []
  if predict_cumulants:
    aux_tasks.append(
      CumulantsFromMemoryAuxTask(
        [config.cumulant_hidden_size]*config.cumulant_layers + [state_dim],
        normalize=config.normalize_cumulants,
        activation=config.cumulant_act,
        construction=config.cumulant_const))

  net = BasicRecurrent(
    inputs_prep_fn=convert_floats,
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

def build_usfa_farm_head(config, state_dim, num_actions, farm_memory, sf_input_fn=None, flat=True, Cls=UsfaHead):

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

  usfa_head = Cls(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      head_layers=config.out_q_layers,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      z_as_train_task=config.z_as_train_task,
      sf_input_fn=sf_input_fn,
      multihead=config.multihead,
      concat_w=config.concat_w,
      task_embed=task_embed,
      normalize_task=config.normalize_task and config.embed_task,
      )
  return usfa_head

def usfa_farmflat_model(config, env_spec, predict_cumulants=True, learn_model=True, **net_kwargs):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.task.shape[0]

  farm_memory = build_farm(config, return_attn=True)

  usfa_head = build_usfa_farm_head(
    config=config,
    state_dim=state_dim,
    num_actions=num_actions,
    farm_memory=farm_memory,
    sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
    flat=True)

  aux_tasks = []
  if learn_model:
    # takes structured farm input
    aux_tasks.append(
      FarmModel(
        output_sizes=max(config.model_layers-1, 0)*[config.module_size],
        num_actions=num_actions,
        seperate_params=config.seperate_model_params,
        # activation=getattr(jax.nn, config.activation)
        ),
      )

  if predict_cumulants:
    # takes structured farm input
    if config.seperate_cumulant_params:
      cumulants_per_module = state_dim//farm_memory.nmodules
      aux_tasks.append(
        FarmIndependentCumulants(
          activation=config.cumulant_act,
          module_cumulants=cumulants_per_module,
          hidden_size=config.cumulant_hidden_size,
          layers=config.cumulant_layers,
          seperate_params=config.seperate_cumulant_params,
          construction=config.cumulant_const,
          normalize_delta=config.normalize_delta and getattr(config, "contrast_module_coeff", 0) > 0,
          normalize_state=getattr(config, "contrast_time_coeff", 0) > 0,
          normalize_cumulants=config.normalize_cumulants
          ))
    else:
      aux_tasks.append(
        FarmCumulants(
              activation=config.cumulant_act,
          module_cumulants=usfa_head.out_dim,
          hidden_size=config.cumulant_hidden_size,
          layers=config.cumulant_layers,
          aggregation='concat',
          normalize_cumulants=config.normalize_cumulants,
          normalize_delta=config.normalize_delta,
          construction=config.cumulant_const,
          ))

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
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=farm_memory,
    prediction_prep_fn=prediction_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=usfa_head.evaluation,
    aux_tasks=aux_tasks,
    **net_kwargs
  )


def build_msf_head(config, state_dim, num_actions):
  if config.sf_net == "flat":

    head = UsfaHead(
        num_actions=num_actions,
        state_dim=state_dim,
        hidden_size=config.out_hidden_size,
        head_layers=config.out_q_layers,
        policy_size=config.policy_size,
        variance=config.variance,
        nsamples=config.npolicies,
        duelling=config.duelling,
        policy_layers=config.policy_layers,
        z_as_train_task=config.z_as_train_task,
        sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
        multihead=config.multihead,
        concat_w=config.concat_w,
        task_embed=0,
        normalize_task=config.normalize_task and config.embed_task,
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
    if config.sf_net == "independent":
      relational_net = lambda x: x
    elif config.sf_net == "relational":
      position_embed=config.embed_position if not config.position_hidden else 0
      relational_net = RelationalNet(
        layers=config.sf_net_layers,
        num_heads=config.sf_net_heads,
        attn_size=config.sf_net_attn_size,
        layernorm=config.layernorm_rel,
        pos_mlp=config.resid_mlp,
        residual=config.relate_residual,
        position_embed=position_embed,
        w_init_scale=config.relate_w_init,
        res_w_init_scale=config.resid_w_init,
        init_bias=config.relate_b_init,
        relu_gate=config.res_relu_gate,
        shared_parameters=not config.seperate_value_params)
    else:
      raise NotImplementedError(config.sf_net)

    cumulants_per_module = state_dim//config.nmodules
    head = FarmUsfaHead(
          num_actions=num_actions,
          cumulants_per_module=cumulants_per_module,
          nmodules=config.nmodules,
          share_output=config.sf_share_output,
          hidden_size=config.out_hidden_size,
          head_layers=config.out_q_layers,
          policy_size=config.policy_size,
          variance=config.variance,
          nsamples=config.npolicies,
          relational_net=relational_net,
          policy_layers=config.policy_layers,
          struct_policy=config.struct_policy_input,
          eval_task_support=config.eval_task_support,
          argmax_mod=config.argmax_mod,
          multihead=config.seperate_value_params, # seperate params per cumulants
          vmap_multihead=config.farm_vmap,
          )
    def pred_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_prep_fn(inputs=inputs, memory_out=memory_out.hidden)

    def eval_prep_fn(inputs, memory_out, *args, **kwargs):
      """Concat Farm module-states before passing them."""
      return usfa_eval_prep_fn(inputs=inputs, memory_out=memory_out.hidden)

  return head, pred_prep_fn, eval_prep_fn


def build_msf_phi_net(config, module_cumulants):
  contrast_module = getattr(config, "contrast_module_coeff", 0) > 0
  contrast_module_delta = getattr(config, "contrast_module_pred", 'delta') == 'delta'
  contrast_module_state = getattr(config, "contrast_module_pred", 'delta') == 'state'
  contrast_time = getattr(config, "contrast_time_coeff", 0) > 0

  normalize_delta = config.normalize_delta and contrast_module and contrast_module_delta
  normalize_state = config.normalize_state and (contrast_module_state or contrast_time)


  if config.phi_net == "flat":
    if config.sf_net == "flat":
      module_cumulants = module_cumulants
    else: # modular
      module_cumulants = module_cumulants*config.nmodules
    return FarmCumulants(
          activation=config.cumulant_act,
          module_cumulants=module_cumulants,
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

    phi_net =  FarmIndependentCumulants(
        activation=config.cumulant_act,
        module_cumulants=module_cumulants,
        hidden_size=config.cumulant_hidden_size,
        layers=config.cumulant_layers,
        seperate_params=config.seperate_cumulant_params,
        construction=config.cumulant_const,
        relational_net=relational_net,
        normalize_delta=normalize_delta,
        normalize_state=normalize_state,
        normalize_cumulants=config.normalize_cumulants)

    return phi_net
  raise RuntimeError

def msf(config, env_spec, predict_cumulants=True, learn_model=False, **net_kwargs):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.task.shape[0]

  module_size = config.module_size
  if config.module_task_dim != 0:
    config.nmodules = int(state_dim//config.module_task_dim)
  if config.module_size is None:
    config.module_size = config.memory_size//config.nmodules
  config.memory_size = config.nmodules*config.module_size

  farm_memory = build_farm(config, return_attn=True)

  assert config.sf_net in ['flat', 'independent', 'relational', 'relational']
  assert config.phi_net in ['flat', 'independent', 'relational']

  usfa_head, pred_prep_fn, eval_prep_fn = build_msf_head(config, state_dim, num_actions)

  aux_tasks = []
  learn_model = learn_model and (getattr(config, "contrast_time_coeff", 0) > 0 or getattr(config, "contrast_module_coeff", 0) > 0)

  if learn_model:
    # takes structured farm input
    aux_tasks.append(
      FarmModel(
        output_sizes=max(config.model_layers-1, 0)*[config.module_size],
        num_actions=num_actions,
        seperate_params=config.seperate_model_params,
        ),
      )

  if predict_cumulants:
    # takes structured farm input
    phi_net = build_msf_phi_net(config, usfa_head.cumulants_per_module)
    aux_tasks.append(phi_net)

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=farm_memory,
    prediction_prep_fn=pred_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=eval_prep_fn,
    evaluation=usfa_head.evaluation,
    aux_tasks=aux_tasks,
    **net_kwargs
  )
