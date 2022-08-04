"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/kitchen_combo/goto_distributed.py \
    --agent r2d1
"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from absl import app
from absl import flags
import acme
from acme.utils import paths
import functools
from pprint import pprint

from agents import td_agent
from utils import make_logger, gen_log_dir
from utils import data as data_utils

from projects.common.train_distributed import build_common_program
from projects.common.observers import LevelReturnObserver


FLAGS = flags.FLAGS

def build_program(
  agent: str,
  env: str,
  num_actors : int,
  save_config_dict: dict=None,
  wandb_init_kwargs=None,
  update_wandb_name=True, # use path from logdir to populate wandb name
  env_kwargs=None,
  group='experiments', # subdirectory that specifies experiment group
  hourminute=True, # whether to append hour-minute to logger path
  log_every=5.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None, # where to save everything
  debug: bool=False,
  **kwargs,
  ):
  env_kwargs = env_kwargs or dict()
  if env == "kitchen_combo":
    from projects.kitchen_combo import combo_helpers

    setting = env_kwargs.get('setting', 'medium')
    # -----------------------
    # load env stuff
    # -----------------------
    environment_factory = lambda is_eval: combo_helpers.make_environment(
        evaluation=is_eval, path=path, setting=setting)
    env = environment_factory(False)
    env_spec = acme.make_environment_spec(env)
    del env
    # -----------------------
    # load agent/network stuff
    # -----------------------
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = combo_helpers.load_agent_settings(agent, env_spec, config_kwargs)

  elif env == "fruitbot":
    from projects.kitchen_combo import fruitbot_helpers
    print("="*20,'env kwargs', "="*20)
    pprint(env_kwargs)
    setting = env_kwargs.get('setting', 'taskgen_long_easy')
    # -----------------------
    # load env stuff
    # -----------------------
    environment_factory = lambda is_eval: fruitbot_helpers.make_environment(
        evaluation=is_eval,
        **env_kwargs)
    env = environment_factory(False)
    env_spec = acme.make_environment_spec(env)
    del env
    # -----------------------
    # load agent/network stuff
    # -----------------------
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = fruitbot_helpers.load_agent_settings(agent, env_spec, config_kwargs)
    try:
      if config.eval_task_support is None:
        if 'procgen' in setting:
          config.eval_task_support = 'eval'
        elif 'taskgen' in setting:
          config.eval_task_support = 'train'
    except AttributeError as e:
      pass

  elif env == "minihack":
    from projects.kitchen_combo import minihack_helpers
    print("="*20,'env kwargs', "="*20)
    pprint(env_kwargs)
    # -----------------------
    # load env stuff
    # -----------------------
    setting = env_kwargs.get('setting', 'room_small')
    environment_factory = lambda is_eval: minihack_helpers.make_environment(
        evaluation=is_eval,
        **env_kwargs)
    env = environment_factory(False)
    env_spec = acme.make_environment_spec(env)
    del env
    # -----------------------
    # load agent/network stuff
    # -----------------------
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = minihack_helpers.load_agent_settings(agent, env_spec, config_kwargs)
  else:
    raise NotImplementedError(FLAGS.env)



  if debug:
      config.batch_size = 32
      config.burn_in_length = 0
      config.trace_length = 80 # shorter
      config.sequence_period = 40
      config.bootstrap_n = 40
      # config.prefetch_size = 0
      # config.samples_per_insert_tolerance_rate = 0.1
      # config.samples_per_insert = 0.0 # different
      # config.num_parallel_calls = 1
      config.min_replay_size = 1_000 # smaller
      config.max_replay_size = 10_000 # smaller
      kwargs['colocate_learner_replay'] = False

  # -----------------------
  # define dict to save. add some extra stuff here
  # -----------------------
  save_config_dict = save_config_dict or dict()
  save_config_dict.update(config.__dict__)
  save_config_dict.update(
    agent=agent,
    setting=setting,
    group=group
  )

  # -----------------------
  # data stuff:
  #   construct log directory if necessary
  #   + observer of data
  # -----------------------
  if not log_dir:
    log_dir, config_path_str = gen_log_dir(
      base_dir=f"{path}/results/{FLAGS.env}/distributed/{group}",
      hourminute=hourminute,
      return_kwpath=True,
      seed=config.seed,
      agent=str(agent))

    if wandb_init_kwargs and update_wandb_name:
      wandb_init_kwargs['name'] = config_path_str

  observers = [LevelReturnObserver()]
  # -----------------------
  # wandb settup
  # -----------------------
  os.chdir(path)
  return build_common_program(
    environment_factory=environment_factory,
    env_spec=env_spec,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    config=config,
    NetworkCls=NetworkCls,
    NetKwargs=NetKwargs,
    LossFn=LossFn,
    num_evaluators=2,
    LossFnKwargs=LossFnKwargs,
    num_actors=num_actors,
    save_config_dict=save_config_dict,
    log_every=log_every,
    observers=observers,
    **kwargs,
    )

def main(_):
  config_kwargs=dict(seed=FLAGS.seed)

  if FLAGS.max_number_of_steps is not None:
    config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps

  wandb_init_kwargs=dict(
    project=FLAGS.wandb_project,
    entity=FLAGS.wandb_entity,
    group=FLAGS.group if FLAGS.group else FLAGS.agent, # organize individual runs into larger experiment
    notes=FLAGS.notes,
  )

  program = build_program(
    agent=FLAGS.agent,
    env=FLAGS.env,
    num_actors=FLAGS.num_actors,
    config_kwargs=config_kwargs,
    wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
    debug=FLAGS.debug,
    )

  # Launch experiment.
  controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='-1')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='-1'))}
  )
  controller.wait()

if __name__ == '__main__':
  app.run(main)
