"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments.

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
from experiments.common.train import run
from utils import make_logger, gen_log_dir
from experiments.common.observers import LevelReturnObserver, LevelAvgReturnObserver

FLAGS = flags.FLAGS


def main(_):
  config = dict()
  if FLAGS.test:
    config['max_replay_size'] = 10_000
    config['min_replay_size'] = 10

    from pprint import pprint
    print("="*50)
    print("="*20, "testing", "="*20)
    pprint(config)
    print("="*50)

  if FLAGS.env == "goto":
    from experiments.iclr2023 import borsa_helpers
    env = borsa_helpers.make_environment(
      setting=FLAGS.env_setting,
      evaluation=FLAGS.evaluate)

    env_spec = acme.make_environment_spec(env)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = borsa_helpers.load_agent_settings(FLAGS.agent, env_spec,
      config_kwargs=config)

  elif FLAGS.env == "fruitbot":
    from experiments.iclr2023 import fruitbot_helpers
    setting = FLAGS.env_setting or 'taskgen_long_easy'
    env_kwargs=dict(
      setting=setting,
      max_episodes=4,
      completion_bonus=0.0,
      env_reward_coeff=10.0,
      env_task_dim=4,
    )
    env = fruitbot_helpers.make_environment(
      **env_kwargs,
      evaluation=FLAGS.evaluate)
    env_spec = acme.make_environment_spec(env)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = fruitbot_helpers.load_agent_settings(FLAGS.agent, env_spec, config_kwargs=config)

    try:
      if config.eval_task_support is None:
        if 'procgen' in setting:
          config.eval_task_support = 'eval'
        elif 'taskgen' in setting:
          config.eval_task_support = 'train'
        else:
          raise RuntimeError(setting)
    except AttributeError as e:
      print(e)

  elif FLAGS.env == "minihack":
    from experiments.iclr2023 import minihack_helpers
    setting = FLAGS.env_setting or ''
    env_kwargs=dict(
      setting=setting,
      # max_episodes=4,
      # completion_bonus=0.0,
    )
    env = minihack_helpers.make_environment(
      **env_kwargs,
      evaluation=FLAGS.evaluate)
    env_spec = acme.make_environment_spec(env)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = minihack_helpers.load_agent_settings(FLAGS.agent, env_spec, config_kwargs=config)

  else:
    raise NotImplementedError(FLAGS.env)

  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir=f"results/{FLAGS.env}/local",
    agent=FLAGS.agent,
    seed=config.seed)

  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=FLAGS.group if FLAGS.group else FLAGS.agent, # organize individual runs into larger experiment
    notes=FLAGS.notes,
  )

  observers = [LevelAvgReturnObserver(reset=200)]
  run(
    env=env,
    env_spec=env_spec,
    config=config,
    NetworkCls=NetworkCls,
    NetKwargs=NetKwargs,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    log_dir=log_dir,
    evaluate=FLAGS.evaluate,
    seed=FLAGS.seed,
    observers=observers,
    num_episodes=FLAGS.num_episodes,
    wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
    init_only=FLAGS.init_only,
    )


if __name__ == '__main__':
  app.run(main)
