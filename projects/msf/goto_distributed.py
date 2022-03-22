"""
Run Successor Feature based agents and baselines on 
  BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python projects/msf/goto_distributed.py \
    --agent r2d1
"""

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


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
from utils import data as data_utils


# -----------------------
# flags
# -----------------------
flags.DEFINE_string('experiment', None, 'experiment_name.')
flags.DEFINE_string('agent', 'r2d1', 'which agent.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_actors', 1, 'Number of actors.')
flags.DEFINE_integer('max_number_of_steps', None, 'Maximum number of steps.')
flags.DEFINE_bool('wandb', False, 'whether to log.')

FLAGS = flags.FLAGS

def build_program(
  agent,
  num_actors,
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
  environment_factory = lambda is_eval: helpers.make_environment(
    evaluation=is_eval, path=path, setting=setting)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  # -----------------------
  # load agent/network stuff
  # -----------------------
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(agent, env_spec, config_kwargs, setting=setting)


  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      eval_network=True)

  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn, LossFnKwargs=LossFnKwargs,
    )


  # -----------------------
  # loggers
  # -----------------------
  agent = str(agent)
  extra = dict(seed=config.seed)
  if experiment:
    extra['exp'] = experiment
  log_dir = log_dir or gen_log_dir(
    base_dir=f"{path}/results/msf/distributed",
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

  # -----------------------
  # wandb setup
  # -----------------------
  def wandb_wrap_logger(logger_fn):
    def make_logger(*args, **kwargs):
      import wandb
      # TODO: fix ugly hack
      date, settings = log_dir.split("/")[-3: -1]
      # wandb.init(project="msf", group=f"{date}/{settings}", entity="wcarvalho92")
      wandb.init(
            project='msf',
            entity="wcarvalho92",
            # dir=path,
            name=f"{date}/{settings}",
            group=f"{date}/{settings}",
            # name='Gridworld few tasks: policy and value; a2c outer loss; rmsprop theta, eta optim; update mu once per 10 theta, eta updates',

            # name='Gridworld One task; a2c outer loss; rmsprop train theta, eta, mu; unroll=4; slow mu lr',

            # notes='nGVF={}; theta lr={}, eps={}; eta lr={}, eps={}, entropy_reg={}'.format(
            #     config['num_gvfs'], 
            #     config['a2c_opt_kwargs']['learning_rate'], config['a2c_opt_kwargs']['eps'], 
            #     # config['eta_opt_kwargs']['eta_learning_rate'], config['eta_opt_kwargs']['eps'],
            #     config['eta_opt_kwargs']['learning_rate'], config['eta_opt_kwargs']['eps'],
            #     config['entropy_coef'], 
            # ), 
            save_code=False,
            config=config.__dict__,
        )
      return logger_fn(*args, **kwargs)
    return make_logger

  if use_wandb:
    os.chdir(path)

    import wandb
    wandb.config = config.__dict__
    date, settings = log_dir.split("/")[-3: -1]
    wandb.init(
            project='msf',
            entity="wcarvalho92",
            # dir=path,
            name=f"{date}/{settings}",
            group=f"{date}/{settings}",
            save_code=False,
            config=config.__dict__,
        )

    logger_fn = wandb_wrap_logger(logger_fn)
    actor_logger_fn = wandb_wrap_logger(actor_logger_fn)
    evaluator_logger_fn = wandb_wrap_logger(evaluator_logger_fn)

  # -----------------------
  # save config
  # -----------------------
  paths.process_path(log_dir)
  config_path = os.path.join(log_dir, 'config.json')
  data_utils.save_dict(
    file=config_path,
    dictionary=config.__dict__,
    agent=agent,
    setting=setting,
    experiment=experiment)

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
