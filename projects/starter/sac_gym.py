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

"""Example running SAC in JAX on the OpenAI Gym."""

import launchpad as lp
from absl import app
from absl import flags
from acme.agents.jax import sac
from acme import wrappers

import dm_env
import gym


FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'MountainCarContinuous-v0',
                    'GYM environment task (str).')

def make_environment(
    evaluation: bool = False,
    task: str = 'MountainCarContinuous-v0') -> dm_env.Environment:
  """Creates an OpenAI Gym environment."""
  del evaluation

  # Load the gym environment.
  environment = gym.make(task)

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  # Clip the action returned by the agent to the environment spec.
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment

def main(_):
  task = FLAGS.task
  environment_factory = lambda is_eval: make_environment(is_eval, task)
  program = sac.DistributedSAC(
      environment_factory=environment_factory,
      network_factory=sac.make_networks,
      config=sac.SACConfig(**{
        'num_sgd_steps_per_step': 64,
        'batch_size' : 32,
        'min_replay_size' : 1000,
        },),
      num_actors=4,
      seed=1,
      max_number_of_steps=1000000).build()

  # Launch experiment.
  lp.launch(program)


if __name__ == '__main__':
  app.run(main)