"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/msfa/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python -m ipdb -c continue experiments/exploration1/goto.py \
    --agent r2d1_noise

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
from experiments.exploration1 import helpers
from experiments.common.train import run
from utils import make_logger, gen_log_dir
from experiments.common.observers import LevelReturnObserver, LevelAvgReturnObserver

# -----------------------
# flags
# -----------------------
# flags.DEFINE_string('agent', 'r2d1', 'which agent.')
# flags.DEFINE_string('env_setting', 'small', 'which environment setting.')
# flags.DEFINE_integer('num_episodes', int(1e5), 'Number of episodes to train for.')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_bool('test', True, 'whether testing.')
# flags.DEFINE_bool('evaluate', True, 'whether to use evaluation policy.')

# # -----------------------
# # wandb
# # -----------------------
# flags.DEFINE_bool('wandb', False, 'whether to log.')
# flags.DEFINE_string('wandb_project', 'msf_sync', 'wand project.')
# flags.DEFINE_string('wandb_entity', None, 'wandb entity')
# flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
# flags.DEFINE_string('notes', '', 'notes for wandb.')


FLAGS = flags.FLAGS


def main(_):
  env = helpers.make_environment(setting=FLAGS.env_setting, evaluation=FLAGS.evaluate)
  env_spec = acme.make_environment_spec(env)

  config = dict()
  if FLAGS.test:
    config['max_replay_size'] = 10_000
    config['min_replay_size'] = 1_000
    config['seperate_cumulant_params'] = False
    config['seperate_value_params'] = False
    # config['sf_net'] = 'flat'
    # config['phi_net'] = 'flat'
    config['module_size'] = 160
    config['memory_size'] = None
    # config['share_init_bias'] = 1.0
    # config['memory_size'] = 460
    # config['module_attn_heads'] = 2
    # config['grad_period'] = 0
    # config['schedule_end'] = 40e3
    # config['final_lr_scale'] = 1e-1
    print("="*50)
    print("="*20, "testing", "="*20)
    print("="*50)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(FLAGS.agent, env_spec, setting=FLAGS.env_setting, config_kwargs=config)

  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/exploration1/local",
    agent=FLAGS.agent,
    seed=config.seed)

  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=FLAGS.group if FLAGS.group else FLAGS.agent, # organize individual runs into larger experiment
    notes=FLAGS.notes,
  )

  run(
    env=env,
    env_spec=env_spec,
    config=config,
    NetworkCls=NetworkCls,
    NetKwargs=NetKwargs,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    loss_label=loss_label,
    log_dir=log_dir,
    observers = [LevelAvgReturnObserver(reset=100)],
    log_with_key='log_data',
    evaluate=FLAGS.evaluate,
    seed=FLAGS.seed,
    num_episodes=FLAGS.num_episodes,
    wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
    )


if __name__ == '__main__':
  app.run(main)
