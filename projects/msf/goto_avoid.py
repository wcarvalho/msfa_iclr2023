"""Run JAX R2D2 on Atari."""


from absl import app
from absl import flags
import acme
from acme.agents.jax import r2d2
import functools

from acme import wrappers
import dm_env
# import gym

from envs.acme.goto_avoid import GoToAvoid
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper




from utils.wrappers import ObservationRemapWrapper
from agents import td_agent


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS



def make_environment(evaluation: bool = False,
                     tile_size=8,
                     ) -> dm_env.Environment:
  """Loads the Atari environment."""
  env = GoToAvoid(
    tile_size=tile_size,
    obj2rew=dict(
        pan={
            "pan" : 1,
            "plates" : 0,
            "fork" : 0,
            "knife" : 0,
            },
        plates={
            "pan" : 0,
            "plates" : 1,
            "fork" : 0,
            "knife" : 0,
            },
        fork={
            "pan" : 0,
            "plates" : 0,
            "fork" : 1,
            "knife" : 0,
            },
        knife={
            "pan" : 0,
            "plates" : 0,
            "fork" : 0,
            "knife" : 1,
            },
    ),
    wrappers=[functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size)]
    )

  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(
            pickup='state_features',
            mission='task',
            )),
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)

def main(_):
  env = make_environment()
  env_spec = acme.make_environment_spec(env)

  if FLAGS.agent == "r2d1": # Recurretn DQN
    config = td_agent.R2D1Config(
      batch_size=16,
      trace_length=20,
      burn_in_length=10,
      sequence_period=10)

    NetworkCls=td_agent.R2D2Network
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      lstm_size=256,
      hidden_size=128,
      )

    builder=functools.partial(
      td_agent.TDBuilder,
      LossFn=td_agent.R2D2Learning,
      LossFnKwargs=td_agent.r2d2_loss_kwargs(config)
    )

  elif FLAGS.agent == "usfa": # Universal Successor Features
    NetworkCls=td_agent.USFANetwork
    NetKwargs=dict(
      num_actions=env_spec.actions.num_values,
      lstm_size=256,
      hidden_size=128,
      )
    NotImplementedError(FLAGS.agent)
  elif FLAGS.agent == "msf": # Modular Successor Features
    NotImplementedError(FLAGS.agent)
  else:
    raise NotImplementedError(FLAGS.agent)


  agent = td_agent.TDAgent(
      env_spec,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs),
      builder=builder,
      config=config,
      seed=FLAGS.seed,
      )

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)