
import functools
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple

import collections
import acme
import dataclasses
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax
import tree

from acme import core
from acme import specs
from acme import wrappers
from acme.agents.jax import r2d2
from acme.agents.jax.dqn import learning_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.jax.networks import base
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import GymWrapper

from babyai.levels.iclr19_levels import Level_GoToRedBallGrey
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

# ======================================================
# Custom Loss function
# ======================================================

@dataclasses.dataclass
class R2D2Learning(learning_lib.LossFn):
  """R2D2 Learning."""
  discount: float = 0.99
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # More than DQN
  max_replay_size: int = 1_000_000
  store_lstm_state: bool = True
  max_priority_weight: float = 0.9
  bootstrap_n: int = 5
  importance_sampling_exponent: float = 0.2

  burn_in_length: int = None
  clip_rewards : bool = False
  max_abs_reward: float = 1.

  def error(self, data, online_q, online_state, target_q, target_state):
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_q, axis=-1)
    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype)
    rewards = data.reward
    if self.clip_rewards:
      rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)
    rewards = rewards.astype(online_q.dtype)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n,
            tx_pair=self.tx_pair),
        in_axes=1,
        out_axes=1)
    batch_td_error = batch_td_error_fn(
        online_q[:-1],
        data.action[:-1],
        target_q[1:],
        selector_actions[1:],
        rewards[:-1],
        discounts[:-1])
    batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)
    return batch_td_error, batch_loss


  def __call__(
      self,
      network,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key_grad: networks_lib.PRNGKey,
    ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""

    unroll = network.unroll  # convenienve

    # Convert sample data to sequence-major format [T, B, ...].
    data = utils.batch_to_sequence(batch.data)

    # Get core state & warm it up on observations for a burn-in period.
    if self.store_lstm_state:
      # Replay core state.
      online_state = jax.tree_map(lambda x: x[0], data.extras['core_state'])
    else:
      _, batch_size = data.action.shape
      key_grad, key = jax.random.split(key_grad)
      online_state = network.initial_state.apply(params, key, batch_size)
    target_state = online_state

    # Maybe burn the core state in.
    burn_in_length = self.burn_in_length
    if burn_in_length:
      burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
      key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
      _, online_state = unroll.apply(params, key1, burn_obs, online_state)
      key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
      _, target_state = unroll.apply(target_params, key1, burn_obs,
                                     target_state)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    # Unroll on sequences to get online and target Q-Values.

    key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
    online_q, online_state = unroll.apply(params, key1, data.observation, online_state)
    key_grad, key1, key2 = jax.random.split(key_grad, 3) # original code uses key2
    target_q, target_state = unroll.apply(target_params, key1, data.observation,
                               target_state)

    batch_td_error, batch_loss = self.error(data, online_q, online_state, target_q, target_state)

    # Importance weighting.
    probs = batch.info.probability
    importance_weights = (1. / (probs + 1e-6)).astype(online_q.dtype)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    mean_loss = jnp.mean(importance_weights * batch_loss)

    # Calculate priorities as a mixture of max and mean sequence errors.
    abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = (max_priority + mean_priority)

    reverb_update = learning_lib.ReverbUpdate(
        keys=batch.info.key,
        priorities=priorities
        )
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return mean_loss, extra


# ======================================================
# Custom Builder
#  mainly used to redefine learner + loss function
# ======================================================

class TDBuilder(r2d2.R2D2Builder):
  """TD agent Builder. Agent is derivative of R2D2 but may use different network/loss function
  """
  def __init__(self,
               networks: r2d2_networks.R2D2Networks,
               config: r2d2_config.R2D2Config,
               LossFn=R2D2Learning,
               LossFnKwargs=None,
               logger_fn: Callable[[], loggers.Logger] = lambda: None,):
    super().__init__(networks=networks, config=config, logger_fn=logger_fn)
    LossFnKwargs = LossFnKwargs or dict()
    self.loss_fn = LossFn(**LossFnKwargs)

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:

    # The learner updates the parameters (and initializes them).
    return learning_lib.SGDLearner(
        network=networks,
        random_key=random_key,
        optimizer=optax.chain(
            optax.adam(self._config.learning_rate, eps=1e-3),
        ),
        target_update_period=self._config.target_update_period,
        data_iterator=dataset,
        # -----------------------
        # CUSTOM LOSS FUNCTION
        # -----------------------
        loss_fn=self.loss_fn,
        replay_client=replay_client,
        replay_table_name=self._config.replay_table_name,
        counter=counter,
        num_sgd_steps_per_step=4)

# ======================================================
# Types
# ======================================================
# Only simple observations & discrete action spaces for now.
Observation = jnp.ndarray
# Action = int

# initializations
ValueInitFn = Callable[[networks_lib.PRNGKey, Observation, hk.LSTMState],
                             networks_lib.Params]

# calling networks
RecurrentStateFn = Callable[[networks_lib.Params], hk.LSTMState]
ValueFn = Callable[[networks_lib.Params, Observation, hk.LSTMState],
                         networks_lib.Value]


@dataclasses.dataclass
class TDNetworkFns:
  """
  Mainly to enable using learning_lib.SGDLearner.

  Attributes:
    init: Initializes params.
    forward: Computes Q-values using the network at the given recurrent
      state.
    unroll: Applies the unrolled network to a sequence of 
      observations, for learning.
    initial_state: Recurrent state at the beginning of an episode.
  """
  init: ValueInitFn
  forward: ValueFn
  unroll: ValueFn
  initial_state: RecurrentStateFn

# ======================================================
# Custom Agent
#   mainly for use with custom builder
# ======================================================
class TDAgent(local_layout.LocalLayout):
  """Local TD-based learning agent.
  """
  def __init__(
      self,
      spec,
      networks,
      config,
      seed,
      BuilderCls=TDBuilder,
      workdir: Optional[str] = '~/acme',
      counter: Optional[counting.Counter] = None,
      behavior_constructor=r2d2.make_behavior_policy,
  ):
    min_replay_size = config.min_replay_size
    config.samples_per_insert_tolerance_rate = float('inf')
    config.min_replay_size = 1
    super().__init__(
        seed=seed,
        environment_spec=spec,
        # -----------------------
        # custom builder
        # -----------------------
        builder=BuilderCls(networks, config),
        networks=networks,
        policy_network=behavior_constructor(networks, config),
        workdir=workdir,
        min_replay_size=32 * config.sequence_period,
        samples_per_insert=1.,
        batch_size=config.batch_size,
        num_sgd_steps_per_step=config.sequence_period,
        counter=counter,
    )

class DistributedTDAgent(distributed_layout.DistributedLayout):
  """Distributed R2D2 agents from config."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      environment_spec: specs.EnvironmentSpec,
      network_factory: Callable[[specs.EnvironmentSpec], TDNetworkFns],
      config: r2d2.R2D2Config,
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      BuilderCls=TDBuilder,
      behavior_constructor=r2d2.make_behavior_policy,
      # workdir: Optional[str] = '~/acme',
      device_prefetch: bool = False,
      log_to_bigtable: bool = True,
      log_every: float = 10.0,
  ):

    # -----------------------
    # logger fns
    # -----------------------
    logger_fn = functools.partial(loggers.make_default_logger,
      'learner', log_to_bigtable,
      time_delta=log_every, asynchronous=True,
      serialize_fn=utils.fetch_devicearray,
      steps_key='learner_steps')
    actor_logger_fn = distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every)

    # -----------------------
    # builder
    # -----------------------
    td_builder = BuilderCls(
        networks=network_factory(environment_spec),
        config=config,
        logger_fn=logger_fn)

    # -----------------------
    # policy factories
    # -----------------------
    policy_network_factory = (
        lambda n: behavior_constructor(n, config))
    evaluator_policy_network_factory = (
        lambda n: behavior_constructor(n, config, True))


    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=td_builder,
        policy_network=policy_network_factory,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        environment_spec=environment_spec,
        device_prefetch=device_prefetch,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=actor_logger_fn,
        prefetch_size=config.prefetch_size)


# ======================================================
# Network definitions
# ======================================================
class SimpleRecurrentQNetwork(hk.RNNCore):

  """Simple Vanilla Network.
  """
  
  def __init__(self, num_actions: int):
    super().__init__(name='simple_r2d2_network')
    self._embed = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP([50, 50])])
    self._core = hk.LSTM(20)
    self._head = hk.nets.MLP([num_actions])

  def __call__(
      self,
      inputs: jnp.ndarray,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    image = inputs.image
    B = image.shape[0]
    image = image.reshape(B, -1) / 255.0

    embeddings = self._embed(image)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: jnp.ndarray,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image = inputs.image
    T,B = image.shape[:2]
    image = image.reshape(T, B, -1) / 255.0

    embeddings = hk.BatchApply(self._embed)(image)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return q_values, new_states

def make_networks(batch_size, env_spec, NetworkCls, NetKwargs):
  """Builds networks."""

  # ======================================================
  # Functions for use
  # ======================================================
  def forward_fn(x, s):
    model = NetworkCls(**NetKwargs)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = NetworkCls(**NetKwargs)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
    model = NetworkCls(**NetKwargs)
    return model.unroll(inputs, state)

  # Make networks purely functional.
  forward_hk = hk.transform(forward_fn)
  initial_state_hk = hk.transform(initial_state_fn)
  unroll_hk = hk.transform(unroll_fn)

  # ======================================================
  # Define networks init functions.
  # ======================================================
  def initial_state_init_fn(rng, batch_size):
    return initial_state_hk.init(rng, batch_size)
  dummy_obs_batch = utils.tile_nested(
      utils.zeros_like(env_spec.observations), batch_size)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs_batch)

  def unroll_init_fn(rng, initial_state):
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)


  # Make FeedForwardNetworks.
  forward = networks_lib.FeedForwardNetwork(
      init=forward_hk.init, apply=forward_hk.apply)
  unroll = networks_lib.FeedForwardNetwork(
      init=unroll_init_fn, apply=unroll_hk.apply)
  initial_state = networks_lib.FeedForwardNetwork(
      init=initial_state_init_fn, apply=initial_state_hk.apply)

  # create initialization function
  def init(random_key):
    random_key, key_initial_1, key_initial_2 = jax.random.split(random_key, 3)
    initial_state_params = initial_state.init(key_initial_1, batch_size)
    initial_mem_state = initial_state.apply(initial_state_params, key_initial_2, batch_size)
    random_key, key_init = jax.random.split(random_key)
    initial_params = unroll.init(key_init, initial_mem_state)
    return initial_params

  # this conforms to both R2D2 & DQN APIs
  return TDNetworkFns(
      init=init,
      forward=forward,
      unroll=unroll,
      initial_state=initial_state)


# ======================================================
# environments
# ======================================================

# @dataclasses.dataclass
# class BabyAiObs:
#   image: jnp.ndarray
BabyAiObs = collections.namedtuple('BabyAiObs', ('image', ))

class BabyAI(dm_env.Environment):
  """
  """

  def __init__(self):
    """Initializes BabyAI environment."""
    env = Level_GoToRedBallGrey()
    self.gym_env = RGBImgPartialObsWrapper(env, tile_size=12)
    self.env = GymWrapper(self.gym_env)


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    env_obs = self.gym_env.reset()
    image = env_obs['image']
    obs = BabyAiObs(image=image)
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    env_obs, reward, done, info = self.gym_env.step(action)
    image = env_obs['image']
    obs = BabyAiObs(image=image)
    reward = float(reward)

    if done:
      timestep = dm_env.termination(
        reward=reward, observation=obs)
    else:
      timestep = dm_env.transition(
        reward=reward, observation=obs)

    return timestep


  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return self.env.action_spec()

  def observation_spec(self):
    default = self.env.observation_spec() # dict_keys(['image'])
    default = BabyAiObs(**default)
    return default

def make_environment():
  env = BabyAI()
  return wrappers.SinglePrecisionWrapper(env)
