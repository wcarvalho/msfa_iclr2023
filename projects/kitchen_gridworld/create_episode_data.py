"""Run Successor Feature based agents and baselines on 
   BabyAI derivative environments.

Comand I run:
  PYTHONPATH=$PYTHONPATH:. \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python -m ipdb -c continue projects/msf/goto.py \
    --agent r2d1_noise

"""

# Do not preallocate GPU memory for JAX.

import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

import json
from glob import glob
import os.path
from absl import app
from absl import flags
import acme
import functools
import jax

from acme.agents import agent as acme_agent
from acme.jax import savers
from agents import td_agent
from projects.common.train import run
from utils import make_logger, gen_log_dir
from acme.environment_loop import EnvironmentLoop
from acme.jax.layouts import distributed_layout
import numpy as np

from projects.common.train import create_net_prediction_tuple
from projects.kitchen_gridworld import helpers
from projects.kitchen_gridworld import video_utils


# -----------------------
# flags
# -----------------------
# flags.DEFINE_string('agent', 'r2d1', 'which agent.')
# flags.DEFINE_string('env_setting', 'large_respawn', 'which environment setting.')
flags.DEFINE_integer('num_episodes', int(1), 'Number of episodes to evaluate agents for.')
flags.DEFINE_bool('evaluate', True, 'whether to use evaluation policy.')
flags.DEFINE_bool('distributed', False, 'whether to use evaluation policy.')
flags.DEFINE_bool('overwrite', True, 'whether to use evaluation policy.')


FLAGS = flags.FLAGS




def first_path(path_search):
  options = glob(path_search)
  print("Selected", options[0])
  return options[0]

def evaluate(config_kwargs, agent_name, path, num_episodes=5):
  """Changes:

  1. custom observer
  2. custom behavior policy
  3. wrapper to actor

  Args:
      config_kwargs (TYPE): Description
      agent_name (TYPE): Description
      path (TYPE): Description
      num_episodes (int, optional): Description
  """
  env = helpers.make_environment(
    setting=config_kwargs['setting'],
    task_reps=config_kwargs.get('task_reps', 'pickup'),
    evaluation=False # test set (harder)
    )
  max_vocab_size = len(env.env.instr_preproc.vocab) # HACK
  nlevels = len(env.env.all_level_kwargs)
  env_spec = acme.make_environment_spec(env)
  total_episodes=nlevels*num_episodes

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(
    agent=config_kwargs['agent'],
    env_spec=env_spec,
    max_vocab_size=max_vocab_size,
    config_kwargs=config_kwargs)

  # -----------------------
  # prepare networks
  # -----------------------
  PredCls = create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs)
  # insert into global namespace for pickling, etc.
  NetKwargs.update(PredCls=PredCls)


  # -----------------------
  # video utils:
  # -----------------------
  observer = video_utils.DataStorer(path=path, agent=agent_name, seed=config.seed, episodes=total_episodes)

  kwargs={}
  kwargs['behavior_policy_constructor'] = functools.partial(video_utils.make_behavior_policy,
    evaluation=True)

  # -----------------------
  # prepare networks
  # -----------------------
  networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True)

  # -----------------------
  # builder
  # -----------------------
  builder = td_agent.TDBuilder(
    networks=networks,
    config=config,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    learner_kwargs=dict(clear_sgd_cache_period=config.clear_sgd_cache_period)
    )

  # -----------------------
  # learner
  # -----------------------
  key = jax.random.PRNGKey(config.seed)
  learner_key, key = jax.random.split(key)
  learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=None,
        replay_client=None,
        counter=None)

  # -----------------------
  # load checkpoint
  # -----------------------
  dirs = glob(os.path.join(path, "2022*")); dirs.sort()
  ckpt_config= distributed_layout.CheckpointingConfig(
    directory=dirs[-1],
    add_uid=False,
    max_to_keep=None)
  checkpointer = savers.CheckpointingRunner(
          learner,
          key='learner',
          subdirectory='learner',
          # enable_checkpointing=False,
          **vars(ckpt_config)
          )

  # -----------------------
  # actor
  # -----------------------
  policy_network = video_utils.make_behavior_policy(networks, config, evaluation=True)
  actor_key, key = jax.random.split(key)
  actor = builder.make_actor(
      actor_key, policy_network, variable_source=learner)

  # -----------------------
  # make env + run
  # -----------------------
  actor = video_utils.ActorStorageWrapper(
    agent=actor,
    networks=networks,
    observer=observer,
    epsilon=config.evaluation_epsilon,
    seed=config.seed)

  agent = acme_agent.Agent(actor=actor, learner=learner,
    min_observations=0,
    observations_per_step=1)

  loop = EnvironmentLoop(
    env,
    agent,
    should_update=False,
    observers=[observer]
    )
  loop.run(total_episodes)

def evaluate_distributed(config_kwargs, agent_name, path, root_path='.', cuda_idx=0, num_episodes=30, overwrite=True):
  """Changes

  1. custom observer
  2. custom behavior policy
  3. wrapper to actor

  Args:
      config_kwargs (TYPE): Description
      agent_name (TYPE): Description
      path (TYPE): Description
      root_path (str, optional): Description
      cuda_idx (int, optional): Description
      num_episodes (int, optional): Description
  
  Returns:
      TYPE: Description
  """

  environment_factory = lambda is_eval: helpers.make_environment(
    evaluation=False,
    path=root_path,
    setting=config_kwargs['setting'])
  env = environment_factory(False)
  max_vocab_size = len(env.env.instr_preproc.vocab) # HACK
  nlevels = len(env.env.all_level_kwargs)
  env_spec = acme.make_environment_spec(env)
  del env



  config_kwargs['min_replay_size'] = 10_000
  config_kwargs['max_replay_size'] = 100_000
  config_kwargs['samples_per_insert'] = 0.0
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(config_kwargs['agent'], env_spec, config_kwargs=config_kwargs)

  # -----------------------
  # observer
  # -----------------------
  total_episodes = num_episodes*nlevels
  observer = video_utils.DataStorer(
            path=path,
            exit=True,
            agent=agent_name,
            seed=config_kwargs['seed'],
            episodes=total_episodes)
  if os.path.exists(observer.results_file):
    if overwrite:
      print("="*50)
      print(f"Overwriting: {observer.results_file}")
      print("="*50)
    else:
      print("="*50)
      print(f"Skipping: {observer.results_file}")
      print("="*50)
      return

  # -----------------------
  # prepare networks
  # -----------------------
  PredCls = create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs)
  # insert into global namespace for pickling, etc.
  NetKwargs.update(PredCls=PredCls)

  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn,
      LossFnKwargs=LossFnKwargs,
      learner_kwargs=dict(clear_sgd_cache_period=config.clear_sgd_cache_period)
      )

  def network_factory(spec):
    return td_agent.make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      eval_network=True)

  # -----------------------
  # prepare networks
  # -----------------------
  dirs = glob(os.path.join(path, "2022*")); dirs.sort()
  ckpt_config= distributed_layout.CheckpointingConfig(
    directory=dirs[-1],
    add_uid=False,
    max_to_keep=None)



  # custom behavior policy
  evaluator_policy_network_factory = (
      lambda n: video_utils.make_behavior_policy(n, config, True))
  eval_env_factory=lambda key: environment_factory(True)

  # wrapper to actor inside here
  evaluator_factories = [
    video_utils.storage_evaluator_factory(
        environment_factory=eval_env_factory,
        network_factory=network_factory,
        policy_factory=evaluator_policy_network_factory,
        observers=[observer],
        log_to_bigtable=False,
        epsilon=config.evaluation_epsilon,
        seed=config.seed,
        logger_fn=None,
        )
      for _ in range(1)]

  program = td_agent.DistributedTDAgent(
      environment_factory=environment_factory,
      environment_spec=env_spec,
      network_factory=network_factory,
      evaluator_factories=evaluator_factories,
      builder=builder,
      config=config,
      checkpointing_config=ckpt_config,
      seed=config.seed,
      num_actors=0,
      max_number_of_steps=100_000,
      log_every=10000).build()


  controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_THREADING,
    terminal='current_terminal',
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='-1')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='-1'))}
  )
  controller.wait()


def main(_):
  basepath = './results/kitchen_grid/final'
  searches = dict(
      # usfa='baselines3/*usfa,*', # 
      msf='vocab_fix/*', # model, not relational
      # usfa_lstm='baselines3/*usfa_lstm,*', # 
      # uvfa='baselines3/*r2d1*', # 
    )

  for agent, agent_search in searches.items():
    path_search = os.path.join(basepath, agent_search)
    path = first_path(path_search)

    seed_paths = glob(os.path.join(path, "seed=*"))
    seed_paths.sort()

    cuda_idx = 0
    # for seed_path in seed_paths:
    seed_path = seed_paths[0]
    config = first_path(os.path.join(seed_path, "config.json"))
    with open(config, 'r') as f:
      config = json.load(f)

    if FLAGS.distributed:
      evaluate_distributed(
        config_kwargs=config,
        agent_name=agent,
        path=seed_path,
        cuda_idx=cuda_idx%4,
        overwrite=FLAGS.overwrite)
    else:
      evaluate(config_kwargs=config, agent_name=agent, path=seed_path)
    cuda_idx += 1



if __name__ == '__main__':
  app.run(main)
