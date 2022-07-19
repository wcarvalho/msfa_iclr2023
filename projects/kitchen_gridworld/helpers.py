import os.path
import yaml

import acme
import functools

from acme import wrappers
import dm_env
import rlax


from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss
from losses import cumulants
from losses import msfa_stats


from projects.msf.helpers import q_aux_sf_loss
from projects.kitchen_gridworld import nets
from projects.kitchen_gridworld import configs

from envs.acme.tasks_wrapper import TrainTasksWrapper
from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor



# ======================================================
# Environment
# ======================================================
def make_environment(evaluation: bool = False,
                     tile_size=8,
                     room_size=6,
                     num_dists=0,
                     step_penalty=0.0,
                     task_reps='lesslang',
                     max_text_length=10,
                     symbolic=False,
                     path='.',
                     setting=None,
                     struct_and=False,
                     task_reset_behavior='none',
                     debug=False,
                     **kwargs,
                     ) -> dm_env.Environment:
  setting = setting or 'SmallL2NoDist'
  """Loads environments."""

  task_reps_file = f"envs/babyai_kitchen/tasks/task_reps/{task_reps}.yaml"
  task_reps_file = os.path.join(path, task_reps_file)
  assert os.path.exists(task_reps_file)
  with open(task_reps_file, 'r') as f:
    task_reps = yaml.load(f, Loader=yaml.SafeLoader)

  tasks_file = f"envs/babyai_kitchen/tasks/v1/{setting}.yaml"
  tasks_file = os.path.join(path, tasks_file)
  assert os.path.exists(tasks_file)
  
  with open(tasks_file, 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)

  if evaluation and 'test' in tasks:
    task_dicts = tasks['test']
    train_task_dicts = tasks['train']
    env_kwargs=dict(task_reset_behavior=task_reset_behavior)
  else:
    task_dicts = tasks['train']
    train_task_dicts = tasks['train']
    env_kwargs=dict()

  if 'task_reps' in tasks:
    task_reps = tasks['task_reps']

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai_kitchen/vocab.json"))

  env_wrappers = [functools.partial(MissionIntegerWrapper,
        instr_preproc=instr_preproc,
        max_length=max_text_length,
        struct_and=struct_and and evaluation)]

  if not symbolic:
    env_wrappers.append(
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size))

  env = MultitaskKitchen(
    task_dicts=task_dicts,
    tasks_file=tasks,
    tile_size=tile_size,
    path=path,
    num_dists=num_dists,
    task_reps=task_reps,
    step_penalty=step_penalty,
    room_size=room_size,
    wrappers=env_wrappers,
    symbolic=symbolic,
    debug=debug,
    **env_kwargs,
    **kwargs
    )

  # wrappers for dm_env: used by agent/replay buffer
  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(mission='task'))]

  wrapper_list.append(
    functools.partial(TrainTasksWrapper,
        instr_preproc=instr_preproc,
        max_length=max_text_length,
        task_reps=task_reps,
        train_tasks=[t['task_kinds'] for t in train_task_dicts],
      ),
    )

  wrapper_list += [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]



  return wrappers.wrap_all(env, wrapper_list)



# ======================================================
# Building Agent Networks
# ======================================================
def msf(config, env_spec, NetworkCls, NetKwargs=None, use_separate_eval=True, predict_cumulants=True, learn_model=False, task_embedding='none'):

  NetKwargs=NetKwargs or dict(
    config=config,
    env_spec=env_spec,
    predict_cumulants=predict_cumulants,
    learn_model=learn_model,
    task_embedding=task_embedding,
    use_separate_eval=use_separate_eval)

  LossFn = td_agent.USFALearning

  aux_tasks=[
    q_aux_sf_loss(config),
    msfa_stats.MsfaStats()
  ]

  if predict_cumulants:
    aux_tasks.append(
      cumulants.CumulantRewardLoss(
        shorten_data_for_cumulant=True,
        coeff=config.reward_coeff,
        mask_loss=config.phi_mask_loss,
        loss=config.reward_loss,
        l1_coeff=getattr(config, "phi_l1_coeff", 0.0),
        wl1_coeff=getattr(config, "w_l1_coeff", 0.0),
        balance=config.balance_reward))

  cov_coeff = getattr(config, 'cov_coeff', None)

  if cov_coeff is not None:
    aux_tasks.append(
      cumulants.CumulantCovLoss(
        coeff=cov_coeff,
        blocks=config.nmodules,
        loss=config.cov_loss))

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

  LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
  LossFnKwargs.update(
    loss=config.sf_loss,
    mask_loss=config.sf_mask_loss,
    shorten_data_for_cumulant=True, # needed since using delta for cumulant
    extract_cumulants=functools.partial(
        losses.cumulants_from_preds,
        use_target=config.target_phi,
        stop_grad=True,
      ),
    aux_tasks=aux_tasks)

  loss_label = None
  eval_network = False

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network

def load_agent_settings(agent, env_spec, config_kwargs=None, max_vocab_size=30):
  default_config = dict(max_vocab_size=max_vocab_size)
  default_config.update(config_kwargs or {})

  if agent == "r2d1":
  # Recurrent DQN (2.2M params)
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.R2D1Config(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_input='qfn_concat',
      task_embedding='language',
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.r2d1_loss,
      mask_loss=config.q_mask_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_dot":
  # Recurrent DQN (2.2M params)
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.R2D1Config(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)
    config.memory_size = 550

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_input='qfn_dot',
      task_embedding='language',
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.r2d1_loss,
      mask_loss=config.q_mask_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_no_task": 
  # UVFA + noise added to goal embedding
    config = data_utils.merge_configs(
      dataclass_configs=[
      configs.R2D1Config(),
      configs.LangConfig()],
      dict_configs=default_config
    )

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config,
      env_spec=env_spec,
      task_embedding='language',
      task_input='none')
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(loss=config.r2d1_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "modr2d1":
  # Recurrent DQN (2.2M params)
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.ModR2d1Config(),
        configs.FarmConfig(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)

    NetworkCls=nets.modr2d1 # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_embedding='language',
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.r2d1_loss,
      mask_loss=config.q_mask_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning (2.5M params)

    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.USFAConfig(),
        configs.QAuxConfig(),
        configs.RewardConfig(),
        configs.LangConfig(),],
      dict_configs=default_config
      )

    NetworkCls=nets.usfa # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      task_embedding='language',
      use_separate_eval=True,
      predict_cumulants=True)

    aux_tasks=[
        q_aux_sf_loss(config),
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,
          mask_loss=config.phi_mask_loss,
          loss=config.reward_loss,
          l1_coeff=config.phi_l1_coeff,
          wl1_coeff=config.w_l1_coeff,
          balance=config.balance_reward,
          reward_bias=config.step_penalty,
          ),
      ]
    if config.cov_coeff is not None:
      aux_tasks.append(
        cumulants.CumulantCovLoss(coeff=config.cov_coeff, blocks=0.0) # get stats
        )
    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.sf_loss,
      mask_loss=config.sf_mask_loss,
      shorten_data_for_cumulant=True,
      extract_cumulants=functools.partial(
        losses.cumulants_from_preds,
        use_target=config.target_phi,
        stop_grad=True,
      ),
      aux_tasks=aux_tasks)

    loss_label = 'usfa'
    eval_network = config.eval_network

  elif agent == "msf":
  # USFA + cumulants from FARM + Q-learning
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.QAuxConfig(),
        configs.ModularUSFAConfig(),
        configs.RewardConfig(),
        configs.FarmModelConfig(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)

    return msf(
      config,
      env_spec,
      NetworkCls=nets.msf,
      predict_cumulants=True,
      learn_model=True,
      use_separate_eval=True,
      task_embedding='language')

  elif agent == "msf_4mod_small":
  # USFA + cumulants from FARM + Q-learning
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.QAuxConfig(),
        configs.ModularUSFAConfig(),
        configs.RewardConfig(),
        configs.FarmModelConfig(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)
    config.module_size=140
    config.nmodules=4
    config.memory_size=None
    return msf(
      config,
      env_spec,
      NetworkCls=nets.msf,
      predict_cumulants=True,
      learn_model=True,
      use_separate_eval=True,
      task_embedding='language')

  elif agent == "msf_2mod_small":
  # USFA + cumulants from FARM + Q-learning
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.QAuxConfig(),
        configs.ModularUSFAConfig(),
        configs.RewardConfig(),
        configs.FarmModelConfig(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)
    config.module_size=None
    config.nmodules=2
    config.memory_size=460

    return msf(
      config,
      env_spec,
      NetworkCls=nets.msf,
      predict_cumulants=True,
      learn_model=True,
      use_separate_eval=True,
      task_embedding='language')

  elif agent == "msf_monolithic":
  # USFA + cumulants from FARM + Q-learning
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.ModularUSFAConfig(),
        configs.QAuxConfig(),
        configs.RewardConfig(),
        configs.FarmModelConfig(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)
    config.sf_net = 'flat'
    config.phi_net = 'flat'

    return msf(
      config,
      env_spec,
      NetworkCls=nets.msf,
      predict_cumulants=True,
      learn_model=True,
      use_separate_eval=True,
      task_embedding='language')
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
