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

"""Defines distributed and local R2D2 agents, using JAX."""

import functools
from typing import Callable, Optional

from acme import specs
from acme.agents.jax import normalization
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env


from agents.r2d2 import builder
from agents.r2d2 import networks
from agents.r2d2 import config as r2d2_config

NetworkFactory = Callable[[specs.EnvironmentSpec], networks.R2D2Network]


class DistributedR2D2(distributed_layout.DistributedLayout):
  """Distributed program definition for R2D2.
  """

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: NetworkFactory,
      config: r2d2_config.R2D2Config,
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      log_to_bigtable: bool = False,
      log_every: float = 10.0,
      normalize_input: bool = True,
  ):
    logger_fn = functools.partial(loggers.make_default_logger,
                                  'learner', log_to_bigtable,
                                  time_delta=log_every, asynchronous=True,
                                  serialize_fn=utils.fetch_devicearray,
                                  steps_key='learner_steps')
    r2d2_builder = builder.R2D2Builder(config, logger_fn=logger_fn)
    if normalize_input:
      environment_spec = specs.make_environment_spec(environment_factory(False))
      # One batch dimension: [batch_size, ...]
      batch_dims = (0,)
      r2d2_builder = normalization.NormalizationBuilder(
          r2d2_builder,
          environment_spec,
          is_sequence_based=False,
          batch_dims=batch_dims)
    eval_policy_factory = (
        lambda net: networks.apply_policy_and_sample(net, config.eval_epsilon))
    super().__init__(
        seed=seed,
        environment_factory=lambda: environment_factory(False),
        network_factory=network_factory,
        builder=r2d2_builder,
        policy_network=networks.apply_policy_and_sample,
        evaluator_factories=[
            distributed_layout.default_evaluator(
                environment_factory=lambda: environment_factory(True),
                network_factory=network_factory,
                builder=r2d2_builder,
                policy_factory=eval_policy_factory,
                log_to_bigtable=log_to_bigtable)
        ],
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
    )


class R2D2(local_layout.LocalLayout):
  """Local agent for SAC.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.R2D2Network,
      config: r2d2_config.R2D2Config,
      seed: int,
      normalize_input: bool = True,
      counter: Optional[counting.Counter] = None,
  ):
    min_replay_size = config.min_replay_size
    # Local layout (actually agent.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order for deadlocks not to happen we need to
    # disable rate limiting that happens inside the R2D2Builder. This is achieved
    # by the following two lines.
    config.samples_per_insert_tolerance_rate = float('inf')
    config.min_replay_size = 1
    r2d2_builder = builder.R2D2Builder(config)
    if normalize_input:
      # One batch dimension: [batch_size, ...]
      batch_dims = (0,)
      r2d2_builder = normalization.NormalizationBuilder(
          r2d2_builder, spec, is_sequence_based=False, batch_dims=batch_dims)
    self.builder = r2d2_builder
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=r2d2_builder,
        networks=network,
        policy_network=networks.apply_policy_and_sample(network),
        batch_size=config.batch_size,
        samples_per_insert=config.samples_per_insert,
        min_replay_size=min_replay_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
