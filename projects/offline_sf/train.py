"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments."""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.offline_sf import helpers
# from projects.msf.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')

FLAGS = flags.FLAGS


def main(_):
  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/offline_sf/local",
    agent=FLAGS.agent)
  logger_fn = lambda : make_logger(
        log_dir=log_dir, label=f'{FLAGS.agent}')


  # -----------------------
  # environment
  # -----------------------
  env = helpers.make_environment()
  env_spec = acme.make_environment_spec(env)

  import ipdb; ipdb.set_trace()
  print("add ability to set num episodes to make")
  print("add ability save and load data (maybe)")
  dataset = helpers.make_demonstrations(env, FLAGS.batch_size)
  dataset = dataset.as_numpy_iterator()
  print("big chance that data isn't in format expected by learner")

  # -----------------------
  # network
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = helpers.load_agent_settings(
    FLAGS.agent, env_spec, 
    config_kwargs=dict(batch_size=FLAGS.batch_size))


  networks = td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs)

  # -----------------------
  # agent/learner
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
      logger_fn=logger_fn)
  builder = builder(networks, config)

  random_key = jax.random.PRNGKey(FLAGS.seed)

  learner = builder.make_learner(
    random_key=random_key,
    networks=networks,
    dataset=dataset)
  # agent = td_agent.TDAgent(
  #     env_spec,
  #     networks=networks,
  #     builder=builder,
  #     workdir=log_dir,
  #     config=config,
  #     seed=FLAGS.seed,
  #     )

  # # -----------------------
  # # evaluator
  # # -----------------------
  print("create evaluator")
  # evaluator_network = (
  #       lambda n: td_agent.make_behavior_policy(n, config, True))

  # from acme.agents.jax.r2d2 import actor as r2d2_actor
  # # actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
  # #     evaluator_network)
  # actor_initial_state = self._networks.initial_state.apply(
  #       actor_initial_state_params, initial_state_key2, 1)
  # actor_core = r2d2_actor.get_actor_core(
  #   evaluator_network,
  #   actor_initial_state,
  #   self._config.num_epsilons)
  # variable_client = variable_utils.VariableClient(
  #     learner, 'policy', device='cpu')
  # evaluator = actors.GenericActor(
  #     actor_core, key, variable_client, backend='cpu')

  # # def evaluator_network(params: hk.Params, key: jnp.DeviceArray,
  # #                       observation: jnp.DeviceArray) -> jnp.DeviceArray:
  # #   dist_params = network.apply(params, observation)
  # #   return rlax.epsilon_greedy(FLAGS.evaluation_epsilon).sample(
  # #       key, dist_params)


  # -----------------------
  # make env + run
  # -----------------------
  env_logger = make_logger(
    log_dir=log_dir,
    label='environment_loop')

  eval_loop = EnvironmentLoop(
      environment=env,
      actor=evaluator,
      logger=env_logger)

  print("check training loop")
  # Run the environment loop.
  while True:
    # K updates
    for _ in range(FLAGS.evaluate_every):
      agent.update()
    # evaluate
    eval_loop.run(FLAGS.evaluation_episodes)


  # loop = EnvironmentLoop(env, agent, logger=env_logger)
  # loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
