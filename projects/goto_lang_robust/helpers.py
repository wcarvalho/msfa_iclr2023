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


from envs.acme.babyai import BabyAI

# TODO: move each of these to `envs.babyai` and update imports across files
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from envs.babyai_kitchen.wrappers import MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor

# -----------------------
# specific to these set of experiments
# -----------------------
from projects.goto_lang_robust import nets
from projects.goto_lang_robust import configs

def make_environment(evaluation: bool = False,
                     tile_size=12,
                     room_size=6,
                     max_text_length=5, # go to the {color} {type}
                     num_dists=4,
                     instr='goto',
                     path='.',
                     setting: int=2,
                     ) -> dm_env.Environment:
  """Loads environments."""

  # -----------------------
  # setup settings
  # -----------------------
  all_colors=['green', 'red', 'blue', 'purple', 'yellow', 'grey']
  ntest = setting
  ntrain = len(all_colors) - ntest
  key_base = f'{ntest}test_{ntrain}train'
  if evaluation:
    key2colors={
      f'test_{key_base}': all_colors[:ntest],
      f'train_{key_base}': all_colors[ntest:],
    }
  else:
    key2colors={
      f'train_{key_base}': all_colors[ntest:],
    }

  # make word preprocessor for converting language to ints
  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai/vocab.json"))

  env = BabyAI(
    key2colors=key2colors,
    room_size=room_size,
    num_dists=num_dists,
    instr=instr,
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


def load_agent_settings(agent, env_spec, config_kwargs=None, setting='small', max_vocab_size=30):
  default_config = dict(max_vocab_size=max_vocab_size)
  default_config.update(config_kwargs or {})

  if agent == "r2d1": # Recurrent DQN
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1 # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  elif agent == "r2d1_gated": # Recurrent DQN
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1_gated # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  elif agent == "r2d1_noise":
    config = configs.NoiseEnsembleConfig(**default_config)

    NetworkCls=nets.r2d1_noise # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  elif agent == "r2d1_noise_eval":
    config = configs.NoiseEnsembleConfig(**default_config)
    config.eval_network = False # don't use seperate eval network 

    NetworkCls=nets.r2d1_noise # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  elif agent == "r2d1_noise_ensemble":
    config = configs.NoiseEnsembleConfig(**default_config)
    config.loss_coeff = 0 # Turn off main loss

    NetworkCls=nets.r2d1_noise_ensemble # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[
        QLearningEnsembleLoss(
          coeff=1.,
          discount=config.discount)
      ])

  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs
