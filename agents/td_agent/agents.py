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

from acme import core
from acme import environment_loop
from acme import specs
# from acme.agents.jax.r2d2 import builder
from acme.agents.jax import builders
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import rlax
import reverb

from agents.td_agent import configs
from agents.td_agent import builder
from agents.td_agent.utils import make_behavior_policy
from agents.td_agent.types import TDNetworkFns


NetworkFactory = Callable[[specs.EnvironmentSpec], TDNetworkFns]

def evaluator_custom_env_loop(
  environment_factory: distributed_layout.EnvironmentFactory,
  network_factory: distributed_layout.NetworkFactory,
  builder: builders.GenericActorLearnerBuilder,
  policy_factory: distributed_layout.PolicyFactory,
  log_to_bigtable: bool = False,
  EnvLoopCls = environment_loop.EnvironmentLoop,
  logger_fn=None) -> types.EvaluatorFactory:
  """Returns a default evaluator process."""
  def evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment = environment_factory()
    networks = network_factory(specs.make_environment_spec(environment))

    actor = builder.make_actor(
        random_key, policy_factory(networks), variable_source=variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    if logger_fn:
      logger = logger_fn()
    else:
      logger = loggers.make_default_logger(
        'evaluator', log_to_bigtable, steps_key='actor_steps')

    # Create the run loop and return it.
    return EnvLoopCls(environment, actor, counter, logger)
  return evaluator



class DistributedTDAgent(distributed_layout.DistributedLayout):
  """Distributed R2D2 agents from config."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      environment_spec: specs.EnvironmentSpec,
      network_factory: NetworkFactory,
      config: configs.R2D1Config,
      seed: int,
      num_actors: int,
      builder = builder.TDBuilder,
      behavior_policy_constructor=make_behavior_policy,
      max_number_of_steps: Optional[int] = None,
      logger_fn = None,
      actor_logger_fn = None,
      evaluator_logger_fn = None,
      EnvLoopCls = environment_loop.EnvironmentLoop,
      workdir: Optional[str] = '~/acme',
      device_prefetch: bool = False,
      log_to_bigtable: bool = True,
      log_every: float = 10.0,
      multithreading_colocate_learner_and_reverb=True,
      **kwargs,
  ):
    self.EnvLoopCls = EnvLoopCls

    # -----------------------
    # logger fns
    # -----------------------
    logger_fn = logger_fn or functools.partial(loggers.make_default_logger,
      'learner', log_to_bigtable,
      time_delta=log_every, asynchronous=True,
      serialize_fn=utils.fetch_devicearray,
      steps_key='learner_steps')
    actor_logger_fn = actor_logger_fn or distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every)

    # -----------------------
    # builder
    # -----------------------
    td_builder = builder(
        networks=network_factory(environment_spec),
        config=config,
        logger_fn=logger_fn)

    # -----------------------
    # policy factories
    # -----------------------
    policy_network_factory = (
        lambda n: behavior_policy_constructor(n, config))
    evaluator_policy_network_factory = (
        lambda n: behavior_policy_constructor(n, config, True))


    super().__init__(
        seed=seed,
        environment_factory=lambda: environment_factory(False),
        network_factory=network_factory,
        builder=td_builder,
        policy_network=policy_network_factory,
        evaluator_factories=[
            evaluator_custom_env_loop(
                environment_factory=lambda: environment_factory(True),
                network_factory=network_factory,
                builder=td_builder,
                policy_factory=evaluator_policy_network_factory,
                EnvLoopCls=EnvLoopCls,
                log_to_bigtable=log_to_bigtable,
                logger_fn=evaluator_logger_fn)
        ],
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        environment_spec=environment_spec,
        device_prefetch=device_prefetch,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=actor_logger_fn,
        prefetch_size=config.prefetch_size,
        workdir=workdir,
        multithreading_colocate_learner_and_reverb=multithreading_colocate_learner_and_reverb,
        **kwargs)

  def actor(self, random_key: networks_lib.PRNGKey, replay: reverb.Client,
            variable_source: core.VariableSource, counter: counting.Counter,
            actor_id: int) -> environment_loop.EnvironmentLoop:

    """The actor process. 
       Only change from original is to use custom EnvLoopCls."""
    adder = self._builder.make_adder(replay)

    # Create environment and policy core.
    environment = self._environment_factory()

    networks = self._network_factory(specs.make_environment_spec(environment))
    policy_network = self._policy_network(networks)
    actor = self._builder.make_actor(random_key, policy_network, adder,
                                     variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    # Only actor #0 will write to bigtable in order not to spam it too much.
    logger = self._actor_logger_fn(actor_id)
    # Create the loop to connect environment and agent.
    return self.EnvLoopCls(environment, actor, counter, logger)


class TDAgent(local_layout.LocalLayout):
  """Local TD-based learning agent.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      networks: TDNetworkFns,
      config: configs.R2D1Config,
      seed: int,
      builder = builder.TDBuilder,
      behavior_policy_constructor=make_behavior_policy,
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

    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=builder(networks, config),
        networks=networks,
        policy_network=behavior_policy_constructor(networks, config),
        workdir=workdir,
        min_replay_size=32 * config.sequence_period,
        samples_per_insert=1.,
        batch_size=config.batch_size,
        num_sgd_steps_per_step=config.sequence_period,
        counter=counter,
    )
