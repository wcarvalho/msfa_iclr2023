import os.path
import yaml

import acme
import functools

from acme import wrappers
import dm_env


from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses import usfa as usfa_losses
from losses.vae import VaeAuxLoss
from losses.contrastive_model import DeltaContrastLoss
from losses import cumulants
from modules.ensembles import QLearningEnsembleLoss

from projects.comp_babyai import nets
from projects.comp_babyai import configs

from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor


def make_environment(evaluation: bool = False,
                     tile_size=8,
                     max_vocab_size=30,
                     path='.',
                     setting='small_length2_nodist',
                     ) -> dm_env.Environment:
  """Loads environments."""
  settings = dict(
    small_length2_nodist=dict(
      tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=2_no_dist.yaml",
      room_size=6,
      ),
    # medium=dict(
    #   tasks_file="envs/babyai_kitchen/tasks/unseen_arg/length=3_cook.yaml",
    #   room_size=7
    #   )
    )
  settings=settings[setting]
  
  tasks_file = settings['tasks_file']
  with open(os.path.join(path, tasks_file), 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)

  if evaluation:
    task_dicts = tasks['test']
  else:
    task_dicts = tasks['train']

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai_kitchen/vocab.json"))

  env = MultitaskKitchen(
    task_dicts=task_dicts,
    tile_size=tile_size,
    path=path,
    room_size=settings['room_size'],
    wrappers=[ # wrapper for babyAI gym env
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size),
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc,
        max_length=max_vocab_size)],
    )

  # wrappers for dm_env: used by agent/replay buffer
  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(mission='task')),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)


def load_agent_settings(agent, env_spec, config_kwargs=None, setting='small', max_vocab_size=30):
  default_config = dict(max_vocab_size=max_vocab_size)
  default_config.update(config_kwargs or {})

  if agent == "r2d1": # Recurrent DQN
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_farm":

    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config(), configs.FarmConfig()],
      dict_configs=default_config
      )
    NetworkCls=nets.r2d1_farm # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_farm_model":

    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.R2D1Config(), configs.FarmConfig(), configs.FarmModelConfig()],
      dict_configs=default_config
      )

    NetworkCls=nets.r2d1_farm_model # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[DeltaContrastLoss(
                    coeff=config.model_coeff,
                    extra_negatives=config.extra_negatives,
                    temperature=config.temperature,
                  )])

    loss_label = 'r2d1'
    eval_network = config.eval_network

  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network
