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

from acme.jax import savers
from agents import td_agent
from projects.common.train import run
from utils import make_logger, gen_log_dir
from acme.environment_loop import EnvironmentLoop
from acme.jax.layouts import distributed_layout
import numpy as np


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


FLAGS = flags.FLAGS




def first_path(path_search):
  options = glob(path_search)
  print("Selected", options[0])
  return options[0]

def evaluate(config_kwargs, agent_name, path, num_episodes=10):

  env = helpers.make_environment(
    setting=config_kwargs['setting'],
    task_reps=config_kwargs.get('task_reps', 'pickup'),
    evaluation=True # test set (harder)
    )
  max_vocab_size = len(env.env.instr_preproc.vocab) # HACK
  env_spec = acme.make_environment_spec(env)


  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(
    agent=config_kwargs['agent'],
    env_spec=env_spec,
    max_vocab_size=max_vocab_size,
    config_kwargs=config_kwargs)

  # -----------------------
  # agent
  # -----------------------
  builder=functools.partial(td_agent.TDBuilder,
      LossFn=LossFn,
      LossFnKwargs=LossFnKwargs,
      learner_kwargs=dict(clear_sgd_cache_period=config.clear_sgd_cache_period)
      )

  # -----------------------
  # video utils:
  #  1. observer
  #  2. custom behavior policy
  #  3. wrapper to agent
  # -----------------------
  observer = video_utils.DataStorer(path=path, agent=agent_name, seed=config.seed, episodes=1)

  kwargs={}
  kwargs['behavior_policy_constructor'] = functools.partial(video_utils.make_behavior_policy,
    evaluation=True)

  # -----------------------
  # prepare networks
  # -----------------------
  dirs = glob(os.path.join(path, "2022*")); dirs.sort()
  agent = td_agent.TDAgent(
      env_spec,
      networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True),
      builder=builder,
      workdir=dirs[-1],
      config=config,
      seed=config.seed,
      **kwargs,
      )

  ckpt_config= distributed_layout.CheckpointingConfig(
    directory=dirs[-1],
    add_uid=False,
    max_to_keep=None)
  checkpointer = savers.Checkpointer(
          {'learner': agent._learner},
          subdirectory='learner',
          **vars(ckpt_config)
          )

  # -----------------------
  # make env + run
  # -----------------------
  agent._actor = video_utils.ActorStorageWrapper(
    agent=agent._actor,
    observer=observer,
    epsilon=config.evaluation_epsilon,
    seed=config.seed)
  loop = EnvironmentLoop(
    env,
    agent,
    should_update=False,
    observers=[observer]
    )
  loop.run(10)

def evaluate_distributed(config_kwargs, agent_name, path, root_path='.', cuda_idx=0):

  observer = DataDownloadingObserver(
            path=path,
            exit=True,
            agent=agent_name,
            seed=config_kwargs['seed'],
            episodes=num_episodes)

  if os.path.exists(observer.results_file):
    return


  environment_factory = lambda is_eval: helpers.make_environment(
      evaluation=is_eval,
      path=root_path,
      setting=config_kwargs['setting'],
      obj2rew=obj2rew)
  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)
  del env

  config_kwargs['min_replay_size'] = 10_000
  config_kwargs['max_replay_size'] = 100_000
  config_kwargs['samples_per_insert'] = 0.0
  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network = helpers.load_agent_settings(config_kwargs['agent'], env_spec, setting=config_kwargs['setting'], config_kwargs=config_kwargs)

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


  evaluator_factories=None
  evaluator_policy_network_factory = (
      lambda n: td_agent.make_behavior_policy(n, config, True))
  eval_env_factory=lambda key: environment_factory(True)
  evaluator_factories = [
    distributed_layout.default_evaluator_factory(
        environment_factory=eval_env_factory,
        network_factory=network_factory,
        policy_factory=evaluator_policy_network_factory,
        observers=[observer],
        log_to_bigtable=False)
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


  controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
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
      msf='ssf2/*rel*', # model, not relational
      # usfa_lstm='baselines3/*usfa_lstm,*', # 
      # uvfa='baselines3/*r2d1*', # 
    )

  for agent, agent_search in searches.items():
    path_search = os.path.join(basepath, agent_search)
    path = first_path(path_search)

    seed_paths = glob(os.path.join(path, "seed=*"))
    seed_paths.sort()

    cuda_idx = 0
    for seed_path in seed_paths:
      config = first_path(os.path.join(seed_path, "config.json"))
      with open(config, 'r') as f:
        config = json.load(f)

      if FLAGS.distributed:
        evaluate_distributed(
          config_kwargs=config,
          agent_name=agent,
          path=seed_path,
          cuda_idx=cuda_idx%4)
      else:
        evaluate(config_kwargs=config, agent_name=agent, path=seed_path)
      cuda_idx += 1



if __name__ == '__main__':
  app.run(main)
