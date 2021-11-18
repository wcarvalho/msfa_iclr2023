# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN Builder."""
from typing import Callable, Iterator, List, Optional

from functools import partial

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
# from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.dqn import learning_lib
from acme.datasets import reverb as datasets
from acme.jax import utils as jax_utils
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import reverb

from reverb import rate_limiters
import rlax

from agents.r2d2 import actor_core as actor_core_lib
from agents.r2d2 import acting
from agents.r2d2 import config as r2d2_config
from agents.r2d2 import losses
from agents.r2d2 import networks as r2d2_networks

class R2D2Builder(builders.ActorLearnerBuilder):
  """DQN Builder."""

  def __init__(
      self,
      config: r2d2_config.R2D2Config,
      core_state_spec : hk.LSTMState,
      loss_fn: losses.R2D2Learning=None,
      logger_fn: Callable[[], loggers.Logger] = lambda: None,
  ):
    """Creates DQN learner and the behavior policies.

    Args:
      config: DQN config.
      loss_fn: A loss function.
      logger_fn: a logger factory for the learner
    """
    self._config = config
    self._core_state_spec = core_state_spec
    self._sequence_length = config.burn_in_length + config.trace_length + 1

    loss_fn = loss_fn or losses.R2D2Learning
    self._loss_fn = loss_fn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        store_lstm_state=config.store_lstm_state,
        max_priority_weight=config.max_priority_weight,
      )
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Network,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
    return learning_lib.SGDLearner(
        network=networks,
        random_key=random_key,
        optimizer=optax.chain(
            optax.clip_by_global_norm(self._config.max_gradient_norm),
            optax.adam(self._config.learning_rate, eps=1e-3),
        ),
        target_update_period=self._config.target_update_period,
        data_iterator=dataset,
        loss_fn=self._loss_fn,
        replay_client=replay_client,
        replay_table_name=self._config.replay_table_name,
        counter=counter,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        logger=self._logger_fn())

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
    ) -> core.Actor:
    assert variable_source is not None
    # Inference happens on CPU, so it's better to move variables there too.
    variable_client = variable_utils.VariableClient(variable_source, '', device='cpu')

    replay_period = self._config.replay_period
    batch_size = self._config.batch_size
    min_replay_size = self._config.min_replay_size
    samples_per_insert = self._config.samples_per_insert
    return acting.R2D2Actor(
        forward_fn=policy_network.apply,
        initial_state_init_fn=policy_network.init_initial_state,
        initial_state_fn=policy_network.initial_state,
        variable_client=variable_client,
        adder=adder,
        epsilon=self._config.epsilon,
        rng=hk.PRNGSequence(random_key),
        # min_observations = replay_period * max(batch_size, min_replay_size),
        # observations_per_step = (
        #     float(replay_period * batch_size) / samples_per_insert)

      )

  def make_replay_tables(
      self, environment_spec: specs.EnvironmentSpec) -> List[reverb.Table]:

    """Creates reverb tables for the algorithm."""
    if self._config.store_lstm_state:
      extra_spec = self._core_state_spec
    else:
      extra_spec = ()


    return [reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Prioritized(self._config.priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_replay_size,
        rate_limiter=rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders_reverb.SequenceAdder.signature(
            environment_spec, extra_spec, sequence_length=self._sequence_length))]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset iterator to use for learning."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=(
            self._config.batch_size * self._config.num_sgd_steps_per_step),
        prefetch_size=self._config.prefetch_size)
    return dataset.as_numpy_iterator()

  def make_adder(self, replay_client: reverb.Client) -> adders.Adder:
    """Creates an adder which handles observations."""
    return adders_reverb.SequenceAdder(
        client=replay_client,
        sequence_length=self._sequence_length,
        period=self._config.replay_period)
