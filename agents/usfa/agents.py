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
# from acme.agents.jax.r2d2 import builder
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import rlax

from agents.usfa import config as usfa_config
from agents.usfa import utils as usfa_utils
from agents.usfa import builder

NetworkFactory = Callable[[specs.EnvironmentSpec], r2d2_networks.R2D2Networks]


# class DistributedR2D2FromConfig(distributed_layout.DistributedLayout):
#   """Distributed R2D2 agents from config."""

#   def __init__(
#       self,
#       environment_factory: Callable[[bool], dm_env.Environment],
#       environment_spec: specs.EnvironmentSpec,
#       network_factory: NetworkFactory,
#       config: r2d2_config.R2D2Config,
#       seed: int,
#       num_actors: int,
#       workdir: Optional[str] = '~/acme',
#       device_prefetch: bool = False,
#       log_to_bigtable: bool = True,
#       log_every: float = 10.0,
#   ):
#     logger_fn = functools.partial(loggers.make_default_logger,
#                                   'learner', log_to_bigtable,
#                                   time_delta=log_every, asynchronous=True,
#                                   serialize_fn=utils.fetch_devicearray,
#                                   steps_key='learner_steps')
#     r2d2_builder = builder.R2D2Builder(
#         networks=network_factory(environment_spec),
#         config=config,
#         logger_fn=logger_fn)
#     policy_network_factory = (
#         lambda n: r2d2_networks.make_behavior_policy(n, config))
#     evaluator_policy_network_factory = (
#         lambda n: r2d2_networks.make_behavior_policy(n, config, True))
#     super().__init__(
#         seed=seed,
#         environment_factory=lambda: environment_factory(False),
#         network_factory=network_factory,
#         builder=r2d2_builder,
#         policy_network=policy_network_factory,
#         evaluator_factories=[
#             distributed_layout.default_evaluator(
#                 environment_factory=lambda: environment_factory(True),
#                 network_factory=network_factory,
#                 builder=r2d2_builder,
#                 policy_factory=evaluator_policy_network_factory,
#                 log_to_bigtable=log_to_bigtable)
#         ],
#         num_actors=num_actors,
#         environment_spec=environment_spec,
#         device_prefetch=device_prefetch,
#         log_to_bigtable=log_to_bigtable,
#         actor_logger_fn=distributed_layout.get_default_logger_fn(
#             log_to_bigtable, log_every),
#         prefetch_size=config.prefetch_size,
#         workdir=workdir)


# class DistributedR2D2(DistributedR2D2FromConfig):
#   """Distributed R2D2 agent."""

#   def __init__(
#       self,
#       environment_factory: Callable[[bool], dm_env.Environment],
#       environment_spec: specs.EnvironmentSpec,
#       forward: hk.Transformed,
#       unroll: hk.Transformed,
#       initial_state: hk.Transformed,
#       num_actors: int,
#       num_caches: int = 1,
#       burn_in_length: int = 40,
#       trace_length: int = 80,
#       sequence_period: int = 40,
#       batch_size: int = 64,
#       prefetch_size: int = 2,
#       target_update_period: int = 2500,
#       samples_per_insert: float = 0.,
#       min_replay_size: int = 1000,
#       max_replay_size: int = 100_000,
#       importance_sampling_exponent: float = 0.6,
#       priority_exponent: float = 0.9,
#       max_priority_weight: float = 0.9,
#       bootstrap_n: int = 5,
#       clip_rewards: bool = False,
#       tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
#       learning_rate: float = 1e-3,
#       evaluator_epsilon: float = 0.,
#       discount: float = 0.997,
#       variable_update_period: int = 400,
#       seed: int = 1,
#     ):
#     config = r2d2_config.R2D2Config(
#         discount=discount,
#         target_update_period=target_update_period,
#         evaluation_epsilon=evaluator_epsilon,
#         burn_in_length=burn_in_length,
#         trace_length=trace_length,
#         sequence_period=sequence_period,
#         learning_rate=learning_rate,
#         bootstrap_n=bootstrap_n,
#         clip_rewards=clip_rewards,
#         tx_pair=tx_pair,
#         samples_per_insert=samples_per_insert,
#         min_replay_size=min_replay_size,
#         max_replay_size=max_replay_size,
#         batch_size=batch_size,
#         prefetch_size=prefetch_size,
#         importance_sampling_exponent=importance_sampling_exponent,
#         priority_exponent=priority_exponent,
#         max_priority_weight=max_priority_weight,
#     )
#     network_factory = functools.partial(
#         r2d2_networks.make_networks,
#         forward_fn=forward,
#         initial_state_fn=initial_state,
#         unroll_fn=unroll,
#         batch_size=batch_size)
#     super().__init__(
#         seed=seed,
#         environment_factory=environment_factory,
#         environment_spec=environment_spec,
#         network_factory=network_factory,
#         config=config,
#         num_actors=num_actors,
#     )


class TDAgent(local_layout.LocalLayout):
  """Local TD-based learning agent.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      networks: r2d2_networks.R2D2Networks,
      config: usfa_config.Config,
      behavior_policy_constructor, 
      seed: int,
      workdir: Optional[str] = '~/acme',
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

    usfa_builder = builder.USFABuilder(networks, config)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=usfa_builder,
        networks=networks,
        policy_network=usfa_utils.make_behavior_policy(networks, config),
        workdir=workdir,
        min_replay_size=32 * config.sequence_period,
        samples_per_insert=1.,
        batch_size=config.batch_size,
        num_sgd_steps_per_step=config.sequence_period,
        counter=counter,
    )
