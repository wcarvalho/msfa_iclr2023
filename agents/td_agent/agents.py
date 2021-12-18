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

from agents.td_agent import configs
from agents.td_agent import builder
from agents.td_agent.utils import make_behavior_policy
from agents.td_agent.networks import TDNetworkFns


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
