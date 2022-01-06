import os.path
import acme
import functools

from acme import wrappers
from acme.agents.tf.dqfd import bsuite_demonstrations
# import babyai.utils
import dm_env
import json
import tensorflow as tf
import re
import numpy as np

from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.bot import KitchenBot
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper, MissionIntegerWrapper

from utils.wrappers import ObservationRemapWrapper

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
                     tasks=None,
                     room_size=7,
                     partial_obs=False,
                     ) -> dm_env.Environment:
  """Loads environments."""
  tasks = tasks or [
    'pickup',
    'place',
    'heat',
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
    tasks=tasks,
    tile_size=tile_size,
    path=path,
    room_size=room_size,
    wrappers=[
      obs_wrapper,
      functools.partial(MissionIntegerWrapper, instr_preproc=instr_preproc)],
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

# def collect_episode(
#   environment,
#   recorder: DemonstrationRecorder,
#   epsilon=0,
#   ):
#   timestep = environment.reset()
#   while timestep.step_type is not dm_env.StepType.LAST:
#     action = _optimal_deep_sea_policy(environment, timestep)
#     recorder.step(timestep, action)
#     timestep = environment.step(action)
#   recorder.step(timestep, np.zeros_like(action))

# def _nested_stack(sequence: List[Any]):
#   """Stack nested elements in a sequence."""
#   return tree.map_structure(lambda *x: np.stack(x), *sequence)

def _make_dataset(environment, num_optimal : int=10000, num_random : int=3000):
  """Make stochastic demonstration dataset."""

  env = environment.env


  for _ in range(num_optimal):
    obs = env.reset()
    bot = KitchenBot(env)
    obss, actions, rewards = bot.generate_traj()
    obss = [obs]+obss
    obss = np.array(obss)

    import ipdb; ipdb.set_trace()

  for _ in range(num_random):
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    obss = [obs]+obss
    import ipdb; ipdb.set_trace()

  # successes_saved = 0
  # failures_saved = 0
  # while (successes_saved < num_successes) or (failures_saved < num_failures):
  #   # collect_episode(environment, recorder)

  #   if recorder.episode_reward > 0 and successes_saved < num_successes:
  #     recorder.record_episode()
  #     successes_saved += 1
  #   elif recorder.episode_reward <= 0 and failures_saved < num_failures:
  #     recorder.record_episode()
  #     failures_saved += 1
  #   else:
  #     recorder.discard_episode()

  # return recorder.make_tf_dataset()

def make_demonstrations(env: dm_env.Environment,
                        batch_size: int) -> tf.data.Dataset:
  """Prepare the dataset of demonstrations."""

  _make_dataset(env, 10000)

  # recorder = bsuite_demonstrations.DemonstrationRecorder()
  # batch_dataset = bsuite_demonstrations.make_dataset(env, stochastic=False)
  # Combine with demonstration dataset.
  # transition = functools.partial(
  #     _n_step_transition_from_episode, n_step=1, additional_discount=1.)

  dataset = batch_dataset.map(transition)

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
