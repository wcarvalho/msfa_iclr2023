# python3
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

"""USFA Builder."""
from typing import Callable, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb

from agents.usfa.learning import USFALearner

class USFABuilder(r2d2.R2D2Builder):
  """R2D2 Builder.

  This is constructs all of the components for Recurrent Experience Replay in
  Distributed Reinforcement Learning (Kapturowski et al.)
  https://openreview.net/pdf?id=r1lyTjAqYX.
  """

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    # The learner updates the parameters (and initializes them).
    return USFALearner(
        unroll=networks.unroll,
        initial_state=networks.initial_state,
        batch_size=self._config.batch_size,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        optimizer=optax.adam(self._config.learning_rate),
        bootstrap_n=self._config.bootstrap_n,
        tx_pair=self._config.tx_pair,
        prefetch_size=self._config.prefetch_size,
        clip_rewards=self._config.clip_rewards,
        replay_client=replay_client,
        counter=counter,
        logger=self._logger_fn())
