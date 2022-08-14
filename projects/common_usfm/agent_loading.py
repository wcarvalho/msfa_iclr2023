import functools
import rlax

from agents import td_agent
from agents.td_agent import losses
from losses import usfa as usfa_losses
from losses import cumulants
from losses import msfa_stats
from utils import data as data_utils
from projects.common_usfm import nets

def default_loss_kwargs(config):
  return dict(
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      burn_in_length=config.burn_in_length,
      max_replay_size=config.max_replay_size,
      store_lstm_state=config.store_lstm_state,
      max_priority_weight=config.max_priority_weight,
      tx_pair=config.tx_pair,
      bootstrap_n=config.bootstrap_n,
      clip_rewards=config.clip_rewards,
      loss_coeff=config.loss_coeff,
      priority_weights_aux=config.priority_weights_aux,
      priority_use_aux=config.priority_use_aux,
      )

def q_aux_sf_loss(config):
  """Create auxilliary Q-learning loss for SF
  """
  if config.q_aux == "single":
    loss = usfa_losses.QLearningAuxLoss
  elif config.q_aux == "ensemble":
    loss = usfa_losses.QLearningEnsembleAuxLoss
  else:
    raise RuntimeError(config.q_aux)

  if config.sf_loss == 'n_step_q_learning':
    tx_pair = rlax.IDENTITY_PAIR
  elif config.sf_loss == 'transformed_n_step_q_learning':
    tx_pair = rlax.SIGNED_HYPERBOLIC_PAIR
  else:
    raise NotImplementedError(config.sf_loss)


  return loss(
          coeff=config.value_coeff,
          discount=config.discount,
          sched_end=config.q_aux_anneal,
          sched_end_val=config.q_aux_end_val,
          tx_pair=tx_pair,
          mask_loss=config.qaux_mask_loss,
          target_w=config.target_phi,
          elementwise=config.elemwise_qaux_loss,
          stop_w_grad=config.stop_w_grad)

def r2d1(
  env_spec,
  default_config,
  dataclass_configs,
  NetworkCls=None,
  NetKwargs=None,
  LossFnKwargs=None,
  **kwargs):
  """Summary
  
  Args:
      env_spec (TYPE): Description
      default_config (TYPE): Description
      dataclass_configs (None, optional): Description
      NetworkCls (None, optional): Description
      NetKwargs (None, optional): Description
      LossFnKwargs (None, optional): Description
      **kwargs: Description
  
  Deleted Parameters:
      task_input (str, optional): how to input task to network
  
  Returns:
      TYPE: Description
  """
  config = data_utils.merge_configs(
    dataclass_configs=dataclass_configs,
    dict_configs=default_config)

  NetworkCls = NetworkCls or nets.r2d1 # default: 2M params
  NetKwargs =  NetKwargs or dict(
    config=config,
    env_spec=env_spec,
    **kwargs,
    )
  LossFn = td_agent.R2D2Learning
  LossFnKwargs = LossFnKwargs or default_loss_kwargs(config)
  LossFnKwargs.update(
    loss=config.r2d1_loss,
    mask_loss=config.q_mask_loss)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs

def usfa_lstm(
  env_spec,
  default_config : dict,
  dataclass_configs,
  NetworkCls=None,
  NetKwargs=None,
  LossFnKwargs=None,
  predict_cumulants=True,
  **kwargs):
  """Summary
  
  Args:
      env_spec (TYPE): Description
      default_config (dict): dict used to populate config
      dataclass_configs (TYPE): Description
      NetworkCls (None, optional): Description
      NetKwargs (None, optional): Description
      LossFnKwargs (None, optional): Description
      predict_cumulants (bool, optional): Description
      **kwargs: Description
  
  Returns:
      TYPE: Description
  """
  config = data_utils.merge_configs(
    dataclass_configs=dataclass_configs,
    dict_configs=default_config
    )

  NetworkCls = NetworkCls or nets.usfa # default: (1.96M) params
  NetKwargs=dict(
    config=config,
    env_spec=env_spec,
    predict_cumulants=predict_cumulants,
    **kwargs)

  LossFn = td_agent.USFALearning
  LossFnKwargs = default_loss_kwargs(config)

  LossFnKwargs.update(
      loss=config.sf_loss,
      mask_loss=config.sf_mask_loss)


  if predict_cumulants:
    LossFnKwargs['aux_tasks']=[
        q_aux_sf_loss(config),
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,
          mask_loss=config.phi_mask_loss,
          loss=config.reward_loss,
          l1_coeff=config.phi_l1_coeff,
          wl1_coeff=config.w_l1_coeff,
          balance=config.balance_reward,
          elementwise=config.elemwise_phi_loss,
          )
    ]
    LossFnKwargs['shorten_data_for_cumulant']=True

    LossFnKwargs['extract_cumulants'] = functools.partial(
      losses.cumulants_from_preds,
      use_target=config.target_phi,
      stop_grad=True)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs

def msf(
  env_spec,
  default_config : dict,
  dataclass_configs,
  NetworkCls=None,
  NetKwargs=None,
  LossFnKwargs=None,
  predict_cumulants=True,
  learn_model=False,
  **kwargs):
  """Summary
  
  Args:
      env_spec (TYPE): Description
      default_config (dict): dict used to populate config
      dataclass_configs (TYPE): Description
      NetworkCls (None, optional): Description
      NetKwargs (None, optional): Description
      LossFnKwargs (None, optional): Description
      predict_cumulants (bool, optional): Description
      **kwargs: Description
  
  Returns:
      TYPE: Description
  """
  config = data_utils.merge_configs(
    dataclass_configs=dataclass_configs,
    dict_configs=default_config
    )
  assert predict_cumulants, 'never implemented otherwise'

  NetworkCls = NetworkCls or nets.msf
  NetKwargs=dict(
    config=config,
    env_spec=env_spec,
    predict_cumulants=predict_cumulants,
    learn_model=learn_model,
    **kwargs)

  LossFn = td_agent.USFALearning
  LossFnKwargs = default_loss_kwargs(config)


  aux_tasks=[
    q_aux_sf_loss(config),
    msfa_stats.MsfaStats(),
    cumulants.CumulantRewardLoss(
      shorten_data_for_cumulant=True,
      coeff=config.reward_coeff,
      mask_loss=config.phi_mask_loss,
      loss=config.reward_loss,
      l1_coeff=config.phi_l1_coeff,
      wl1_coeff=config.w_l1_coeff,
      balance=config.balance_reward,
      ),
  ]

  if learn_model:
    if config.contrast_module_coeff > 0:
      aux_tasks.append(
          ModuleContrastLoss(
            coeff=config.contrast_module_coeff,
            extra_negatives=config.extra_module_negatives,
            temperature=config.temperature)
          )
    if config.contrast_time_coeff > 0:
      aux_tasks.append(
          TimeContrastLoss(
            coeff=config.contrast_time_coeff,
            extra_negatives=config.extra_time_negatives,
            temperature=config.temperature,
            normalize_step=config.normalize_step)
          )

  LossFnKwargs.update(
    loss=config.sf_loss,
    mask_loss=config.sf_mask_loss,
    shorten_data_for_cumulant=True,
    aux_tasks=aux_tasks)

  LossFnKwargs['extract_cumulants'] = functools.partial(
    losses.cumulants_from_preds,
    use_target=config.target_phi,
    stop_grad=True)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs
