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


from projects.msf.helpers import q_aux_sf_loss
from projects.kitchen_gridworld import nets
from projects.kitchen_gridworld import configs

from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor



# ======================================================
# Environment
# ======================================================
def make_environment(evaluation: bool = False,
                     tile_size=8,
                     room_size=5,
                     num_dists=0,
                     step_penalty=0.0,
                     task_reps='pickup',
                     max_text_length=10,
                     path='.',
                     setting=None,
                     ) -> dm_env.Environment:
  setting = setting or 'SmallL2NoDist'
  """Loads environments."""

  task_rep_options=dict(
    pickup="envs/babyai_kitchen/tasks/task_reps/pickup.yaml",
    lesslang="envs/babyai_kitchen/tasks/task_reps/lesslang.yaml"
    )

  task_reps_file = task_rep_options[task_reps]
  with open(os.path.join(path, task_reps_file), 'r') as f:
    task_reps = yaml.load(f, Loader=yaml.SafeLoader)


  settings = dict(
    EasyPickup=dict(
      tasks_file="envs/babyai_kitchen/tasks/multitask/all_pickup_easy.yaml",
      ),
    SmallL2NoDist=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=2_no_dist.yaml",
      ),
    SmallL2NoDistV2=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=2_no_dist_v2.yaml",
      ),
    SmallL2SliceChill=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=2_slice_chill.yaml",
      ),
    SmallL2Transfer=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=2_slice_chill_clean_transfer.yaml",
      ),
    SmallL2TransferEasy=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=2_slice_chill_clean_transfer_easy.yaml",
      ),
    L2_Args_Multi=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/L2-Args-Multi.yaml",
      ),
    L2_Task_Multi=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/L2-ArgTask-Multi.yaml",
      ),
    L2_Multi=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/L2-Multi.yaml",
      ),
    L2_Multi_Gen=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/L2-Multi-gen-room-dists.yaml",
      ),
    increasingL=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/increasingL.yaml",
      ),
    Clean_Cook_Slice=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/Clean_Cook_Slice.yaml",
      )
    )
  if setting in settings:
    settings=settings[setting]
    tasks_file = settings['tasks_file']
    tasks_file = os.path.join(path, tasks_file)
  else:
    tasks_file = f"envs/babyai_kitchen/tasks/v1/{setting}.yaml"
    tasks_file = os.path.join(path, tasks_file)
    if not os.path.exists(tasks_file):
      raise RuntimeError(f"don't know how to handle setting: {setting}")
  
  with open(tasks_file, 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)

  if evaluation and 'test' in tasks:
    task_dicts = tasks['test']
  else:
    task_dicts = tasks['train']

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai_kitchen/vocab.json"))

  env = MultitaskKitchen(
    task_dicts=task_dicts,
    tile_size=tile_size,
    path=path,
    num_dists=num_dists,
    task_reps=task_reps,
    step_penalty=step_penalty,
    room_size=room_size,
    wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size),
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc,
        max_length=max_text_length)],
    )

  # wrappers for dm_env: used by agent/replay buffer
  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(mission='task')),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)



# ======================================================
# Building Agent Networks
# ======================================================
def msf(config, env_spec, use_seperate_eval=True, predict_cumulants=True, learn_model=False, task_embedding='none'):

  NetworkCls =  nets.msf

  NetKwargs=dict(
    config=config,
    env_spec=env_spec,
    predict_cumulants=predict_cumulants,
    learn_model=learn_model,
    task_embedding=task_embedding,
    use_seperate_eval=use_seperate_eval)

  LossFn = td_agent.USFALearning

  aux_tasks=[q_aux_sf_loss(config)]

  if predict_cumulants:
    nmodules = config.nmodules if config.module_l1 else 1
    aux_tasks.append(
      cumulants.CumulantRewardLoss(
        shorten_data_for_cumulant=True,
        coeff=config.reward_coeff,
        loss=config.reward_loss,
        l1_coeff=config.phi_l1_coeff,
        wl1_coeff=config.w_l1_coeff,
        balance=config.balance_reward,
        reward_bias=config.step_penalty,
        nmodules=nmodules))

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
    shorten_data_for_cumulant=True, # needed since using delta for cumulant
    extract_cumulants=losses.cumulants_from_preds,
    aux_tasks=aux_tasks)

  loss_label = 'usfa'
  eval_network = config.eval_network

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
      task_embedding='language',
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
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
      use_seperate_eval=False,
      predict_cumulants=True)

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.sf_loss,
      shorten_data_for_cumulant=True,
      extract_cumulants=functools.partial(
        losses.cumulants_from_preds,
        stop_grad=True,
      ),
      aux_tasks=[
        q_aux_sf_loss(config),
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,
          loss=config.reward_loss,
          l1_coeff=config.phi_l1_coeff,
          wl1_coeff=config.w_l1_coeff,
          balance=config.balance_reward,
          reward_bias=config.step_penalty,
          ),
      ])

    loss_label = 'usfa'
    eval_network = config.eval_network

  elif agent == "msf":
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

    return msf(
      config,
      env_spec,
      predict_cumulants=True,
      learn_model=True,
      use_seperate_eval=False,
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
      predict_cumulants=True,
      learn_model=True,
      use_seperate_eval=False,
      task_embedding='language')
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
