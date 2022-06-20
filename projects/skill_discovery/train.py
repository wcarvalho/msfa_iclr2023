"""

"""

from absl import app
from absl import flags
import acme
import functools

from agents import td_agent
from projects.common.train import run
from projects.skill_discovery import helpers
from utils import make_logger, gen_log_dir

# -----------------------
# flags
# -----------------------
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('evaluate', False, 'whether to use evaluation policy.')

# -----------------------
# wandb
# -----------------------
flags.DEFINE_bool('wandb', False, 'whether to log.')
flags.DEFINE_bool('init_only', False, 'whether to end after network initialization.')
flags.DEFINE_string('wandb_project', '', 'wand project.')
flags.DEFINE_string('wandb_entity', '', 'wandb username')
flags.DEFINE_string('group', '', 'same as wandb group. way to group runs.')
flags.DEFINE_string('notes', '', 'notes for wandb.')


FLAGS = flags.FLAGS


def main(_):
  env_kwargs=dict(
    room_size=6,
    )
  env = helpers.make_environment(
    setting='pickup',
    evaluation=FLAGS.evaluate, # test set (harder)
    **env_kwargs
    )
  env_spec = acme.make_environment_spec(env)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = helpers.load_agent_settings(
    agent=FLAGS.agent,
    env_spec=env_spec)


  # -----------------------
  # logger
  # -----------------------
  log_dir = gen_log_dir(
    base_dir="results/skill_discovery/local",
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
    log_dir=log_dir,
    evaluate=FLAGS.evaluate,
    seed=FLAGS.seed,
    wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
    debug=False,
    init_only=False,
    pregenerate_named_tuple=False,
    )


if __name__ == '__main__':
  app.run(main)
