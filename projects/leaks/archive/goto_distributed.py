"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess debugging/leaks/goto_distributed.py \
    --agent r2d1_dummy_symbolic --images=False --experiment image

  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess debugging/leaks/goto_distributed.py \
    --agent r2d1_dummy_symbolic --images=False --logging=False --experiment image_logging


  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess debugging/leaks/goto_distributed.py \
    --agent r2d1_dummy_symbolic --images=False --logging=False --experiment learning

  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess debugging/leaks/goto_distributed.py \
    --agent r2d1 --checkpoint=False --sgd=True --batches=False --experiment no_batches_yes_sgd

  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess debugging/leaks/goto_distributed.py \
    --agent r2d1 --checkpoint=False --sgd=False --colocate=True --batches=False --experiment colocate


  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    mprof run --multiprocess debugging/leaks/goto_distributed.py \
    --agent r2d1 --checkpoint=True --sgd=True --colocate=False --batches=True --experiment original_no_static
"""


import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from absl import app
from absl import flags
import acme
from acme.utils import paths
import functools

from agents import td_agent
from projects.msf import helpers
from projects.msf.environment_loop import EnvironmentLoop
from utils import make_logger, gen_log_dir
import pickle


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('experiment', None, 'experiment_name.')
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 1, 'Number of actors.')
flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')
flags.DEFINE_bool('wandb', False, 'whether to log.')


# -----------------------
# debug settingss
# -----------------------
flags.DEFINE_bool('images', True, 'whether to log.')
flags.DEFINE_bool('colocate', False, 'whether to log.')
flags.DEFINE_bool('logging', True, 'whether to log.')
flags.DEFINE_bool('checkpoint', True, 'whether to log.')
flags.DEFINE_bool('sgd', True, 'whether to log.')
flags.DEFINE_bool('batches', True, 'whether to log.')

FLAGS = flags.FLAGS

def build_program(agent, num_actors,
  use_wandb=False,
  setting='small',
  experiment=None,
  log_every=30.0, # how often to log
  config_kwargs=None, # config
  path='.', # path that's being run from
  log_dir=None,
  hourminute=True):
  # -----------------------
  # load env stuff
  # -----------------------
  image_wrapper = FLAGS.images
  environment_factory = lambda is_eval: helpers.make_environment(
    evaluation=is_eval, path=path, setting=setting,
    image_wrapper=image_wrapper)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config_kwargs=dict(
      batch_size=32,
      burn_in_length=0,
      trace_length=20,
      sequence_period=40,
      prefetch_size=0,
      samples_per_insert_tolerance_rate=0.1,
      samples_per_insert=0.0, # single process
      num_parallel_calls=1,
      min_replay_size=1_000,
      max_replay_size=1_500,
      max_number_of_steps=100_000_000
    )
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(agent, env_spec, config_kwargs, setting=setting)

  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      eval_network=True)

  learner_kwargs=dict()
  if FLAGS.checkpoint is not None:
    learner_kwargs['save'] = False

  if FLAGS.sgd is False:
    learner_kwargs['take_sgd_step'] = False

  if FLAGS.batches is False:
    learner_kwargs['cycle_batches'] = False

  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
      learner_kwargs=learner_kwargs,
    )

  # -----------------------
  # loggers
  # -----------------------
  agent = str(agent)
  seed = config.seed
  extra = dict(seed=seed)
  if experiment:
    extra['exp'] = experiment
  log_dir = log_dir or gen_log_dir(
    base_dir=f"{path}/goto/",
    hourminute=hourminute,
    agent=agent,
    **extra)


  logger_fn = lambda : make_logger(
        log_dir=log_dir,
        label=loss_label,
        wandb=use_wandb,
        asynchronous=True)

  actor_logger_fn = lambda actor_id: make_logger(
                  log_dir=log_dir, label='actor',
                  wandb=use_wandb,
                  save_data=actor_id == 0,
                  steps_key="actor_steps",
                  )
  evaluator_logger_fn = lambda : make_logger(
                  log_dir=log_dir, label='evaluator',
                  wandb=use_wandb,
                  steps_key="evaluator_steps",
                  )

  kwargs=dict()
  if FLAGS.logging:
    kwargs.update(
      logger_fn=logger_fn,
      actor_logger_fn=actor_logger_fn,
      evaluator_logger_fn=evaluator_logger_fn,
    )
  # -----------------------
  # build program
  # -----------------------
  colocate = FLAGS.colocate
  workdir = log_dir

  return td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      builder=builder,
      EnvLoopCls=EnvironmentLoop,
      config=config,
      workdir=workdir,
      seed=config.seed,
      num_actors=num_actors,
      max_number_of_steps=config.max_number_of_steps,
      log_every=log_every,
      multithreading_colocate_learner_and_reverb=colocate,
      **kwargs).build()


def main(_):
  config_kwargs=dict(seed=FLAGS.seed)
  if FLAGS.max_number_of_steps is not None:
    config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps
  program = build_program(
    agent=FLAGS.agent,
    num_actors=FLAGS.num_actors,
    experiment=FLAGS.experiment,
    use_wandb=FLAGS.wandb, config_kwargs=config_kwargs)

  # Launch experiment.
  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
  )

if __name__ == '__main__':
  app.run(main)
