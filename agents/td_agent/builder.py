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
from acme.agents.jax.dqn import learning_lib
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

from agents.td_agent.losses import R2D2Learning

class TDBuilder(r2d2.R2D2Builder):
  """TD agent Builder. Agent is derivative of R2D2 but may use different network/loss function
  """
  def __init__(self,
               networks: r2d2_networks.R2D2Networks,
               config: r2d2_config.R2D2Config,
               logger_fn: Callable[[], loggers.Logger] = lambda: None,
               LossFn: learning_lib.LossFn=R2D2Learning,
               LossFnKwargs=None,
               learner_kwargs=None,
               take_sgd_step=True,
               cycle_batches=True):
    super().__init__(networks=networks, config=config, logger_fn=logger_fn)

    LossFnKwargs = LossFnKwargs or dict()
    self.loss_fn = LossFn(**LossFnKwargs)
    self.learner_kwargs = learner_kwargs or dict()

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:

    # The learner updates the parameters (and initializes them).
    logger = self._logger_fn()
    return learning_lib.SGDLearner(
        network=networks,
        random_key=random_key,
        optimizer=optax.chain(
            optax.clip_by_global_norm(self._config.max_gradient_norm),
            optax.adam(self._config.learning_rate, eps=1e-3),
        ),
        target_update_period=self._config.target_update_period,
        data_iterator=dataset,
        loss_fn=self.loss_fn,
        replay_client=replay_client,
        replay_table_name=self._config.replay_table_name,
        counter=counter,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        logger=logger,
        **self.learner_kwargs)
