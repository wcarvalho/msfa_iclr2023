"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python -m ipdb -c continue projects/comp_babyai/train.py \
    --agent r2d1_farm_model
"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # https://github.com/google/jax/issues/8302

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.comp_babyai import helpers
from projects.comp_babyai.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('evaluate', False, 'whether to use evaluation policy.')

FLAGS = flags.FLAGS


def main(_):
  max_vocab_size = 50
  env = helpers.make_environment(max_vocab_size=max_vocab_size)
  env_spec = acme.make_environment_spec(env)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(FLAGS.agent, env_spec, setting=FLAGS.env_setting, max_vocab_size=max_vocab_size)

  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/comp_babyai/local",
    agent=FLAGS.agent,
    seed=config.seed)

  logger_fn = lambda : make_logger(
    log_dir=log_dir, label=loss_label)


  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
      logger_fn=logger_fn)

  kwargs={}
  if FLAGS.evaluate:
    kwargs['behavior_policy_constructor'] = functools.partial(td_agent.make_behavior_policy, evaluation=True)
  agent = td_agent.TDAgent(
      env_spec,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True),
      builder=builder,
      workdir=log_dir,
      config=config,
      seed=FLAGS.seed,
      **kwargs,
      )

  # -----------------------
  # make env + run
  # -----------------------
  env_logger = make_logger(
    log_dir=log_dir,
    label='actor',
    steps_key="steps")

  loop = EnvironmentLoop(env, agent, logger=env_logger)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
