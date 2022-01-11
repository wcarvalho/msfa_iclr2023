"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

To run:
  CUDA_VISIBLE_DEVICES=0 python projects/msf/goto_distributed.py --agent r2d1

"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import launchpad as lp

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.msf import helpers
from projects.msf.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 10, 'Number of actors.')

FLAGS = flags.FLAGS

def build_program(agent, num_actors, 
  log_every=30.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None,
  hourminute=True):
  # -----------------------
  # load env stuff
  # -----------------------
  environment_factory = lambda is_eval: helpers.make_environment(
    evaluation=is_eval, path=path)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = helpers.load_agent_settings(agent, env_spec, config_kwargs)

  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs)

  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
    )


  # -----------------------
  # loggers
  # -----------------------
  agent = str(agent)
  log_dir = log_dir or gen_log_dir(
    base_dir=f"{path}/results/msf/distributed",
    hourminute=hourminute,
    agent=agent,
    seed=config.seed)
  logger_fn = lambda : make_logger(
        log_dir=log_dir, label=agent, asynchronous=True)

  actor_logger_fn = lambda actor_id: make_logger(
                  log_dir=log_dir, label='actor',
                  save_data=actor_id == 0,
                  steps_key="actor_steps",
                  )
  evaluator_logger_fn = lambda : make_logger(
                  log_dir=log_dir,
                  label='evaluator',
                  steps_key="actor_steps",
                  )

  # -----------------------
  # build program
  # -----------------------
  return td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      builder=builder,
      logger_fn=logger_fn,
      actor_logger_fn=actor_logger_fn,
      evaluator_logger_fn=evaluator_logger_fn,
      EnvLoopCls=EnvironmentLoop,
      config=config,
      workdir=log_dir,
      seed=config.seed,
      num_actors=num_actors,
      max_number_of_steps=config.max_number_of_steps,
      log_every=log_every).build()


def main(_):
  program = build_program(FLAGS.agent, FLAGS.num_actors)
  # Launch experiment.
  resources = {
    'actor': lp.PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
    'evaluator': lp.PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
  }
  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
    local_resources=resources,
    terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)
