import os.path
import acme
import functools

from acme import wrappers
from acme.agents.tf.dqfd import bsuite_demonstrations
# import babyai.utils
import dm_env
import jax
import json
import tensorflow as tf
import tree
import re
import numpy as np

from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.bot import KitchenBot
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper, MissionIntegerWrapper

from utils.wrappers import ObservationRemapWrapper
from utils import data

from agents import td_agent
from projects.msf import networks as msf_networks


class InstructionsPreprocessor(object):
  def __init__(self, path):
    if os.path.exists(path):
        # self.vocab = babyai.utils.format.Vocabulary(path)
        self.vocab = json.load(open(path))
    else:
        raise FileNotFoundError(f'No vocab at "{path}"')

  def __call__(self, mission, device=None):
    """Copied from BabyAI
    """
    raw_instrs = []
    max_instr_len = 0


    tokens = re.findall("([a-z]+)", mission.lower())
    instr = np.array([self.vocab[token] for token in tokens])
    raw_instrs.append(instr)
    max_instr_len = max(len(instr), max_instr_len)

    instrs = np.zeros(max_instr_len, dtype=np.int32)
    instrs[:len(instr)] = instr

    return instrs


def make_environment(tile_size=8,
                     path='.',
                     task_kinds=None,
                     room_size=7,
                     partial_obs=False,
                     ) -> dm_env.Environment:
  """Loads environments."""
  task_kinds = task_kinds or [
    'pickup',
    'place',
    # 'heat',
    'pickup_cleaned',
    'pickup_sliced',
    'pickup_chilled',
    'pickup_cooked',
  ]
  if partial_obs:
    obs_wrapper=functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size)
  else:
    obs_wrapper=functools.partial(RGBImgFullyObsWrapper, tile_size=tile_size)

  instr_preproc = InstructionsPreprocessor(
    path="data/babyai_kitchen/vocab.json")

  env = MultitaskKitchen(
    task_kinds=task_kinds,
    tile_size=tile_size,
    path=path,
    room_size=room_size,
    wrappers=[
      obs_wrapper,
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc,
        max_length=30)],
    )

  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(
            # pickup='state_features',
            mission='task',
            )),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)

def collect_optimal_episodes(environment, num_episodes):
  env = environment.env

  episodes = []
  for ep in range(num_episodes):
    obs = env.reset()
    bot = KitchenBot(env)
    obss, actions, rewards, dones = bot.generate_traj()

    # make all same length
    obss = [obs]+obss
    actions.append(0)
    rewards.append(0.)
    dones.append(True)

    # make into numpy arrays
    obss = data.consolidate_dict_list(obss)
    obss = data.dictop(obss, np.array)
    actions, rewards, dones = [np.array(y) for y in [actions, rewards, dones]]

    episodes.append((obss, actions, rewards, dones))

  return episodes

def collect_random_episodes(environment, num_episodes, episode_length=50):
  env = environment.env

  episodes = []
  for ep in range(num_episodes):
    obs = env.reset()
    done = False

    obss = [obs]
    rewards = []
    actions = []
    dones = []
    # collect episode
    for _ in range(episode_length):
      action = env.action_space.sample()
      obs, reward, done, info = env.step(action)
      obss.append(obs)
      actions.append(action)
      rewards.append(reward)
      dones.append(done)

      if done:
        break

    # make into numpy arrays
    obss = data.consolidate_dict_list(obss)
    obss = data.dictop(obss, np.array)
    actions, rewards, dones = [np.array(y) for y in [actions, rewards, dones]]

    episodes.append((obss, actions, rewards, dones))

  return episodes

def _make_dataset(environment, num_optimal : int=2, num_random : int=2):
  """Make demonstration dataset."""

  episodes = collect_optimal_episodes(environment, num_optimal)
  episodes += collect_random_episodes(environment, num_random)


  types = tree.map_structure(lambda x: x.dtype, episodes[0])
  shapes = tree.map_structure(lambda x: tf.TensorShape([None, *x.shape[1:]]), episodes[0])
  dataset = tf.data.Dataset.from_generator(lambda: episodes, types, shapes)
  return dataset


def make_demonstrations(env: dm_env.Environment,
                        batch_size: int,
                        num_optimal : int=30, num_random : int=2) -> tf.data.Dataset:
  """Prepare the dataset of demonstrations."""

  dataset = _make_dataset(env, num_optimal=num_optimal, num_random=num_random)
  dataset = dataset.unbatch()

  # Batch and prefetch.
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset

def load_agent_settings(agent, env_spec, config_kwargs=None):
  config_kwargs = config_kwargs or dict()

  print("="*50)
  print(agent)
  print("="*50)
  if agent == "r2d1": # Recurrent DQN
    config = td_agent.R2D1Config(**config_kwargs)

    NetworkCls=msf_networks.R2D2Network
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      lstm_size=256,
      hidden_size=128,
      )

    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)


  elif agent == "usfa": # Universal Successor Features
    config = td_agent.USFAConfig(**config_kwargs)

    NetworkCls=msf_networks.USFANetwork
    state_dim = env_spec.observations.observation.state_features.shape[0]
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      state_dim=state_dim,
      lstm_size=256,
      hidden_size=128,
      nsamples=config.npolicies,
      )

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)

  # elif agent == "msf": # Modular Successor Features
  else:
    raise NotImplementedError(agent)

  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs
