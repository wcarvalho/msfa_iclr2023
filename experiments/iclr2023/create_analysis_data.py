"""
"""
import os
from absl import app
from absl import flags
import acme
import functools

FLAGS = flags.FLAGS

from experiments.common.create_analysis_data import generate_data
from experiments.iclr2023 import fruitbot_helpers
from experiments.iclr2023 import minihack_helpers
from experiments.iclr2023 import borsa_helpers

flags.DEFINE_string('folder', 'msf2', '')

def load_env_agent_settings(config : dict, setting=None, evaluate=True, 
  env_type='fruitbot'):
  setting = setting or config['setting']

  if env_type == 'fruitbot':
    env = fruitbot_helpers.make_environment(
      setting=setting,
      train_in_eval=False,
      evaluation=evaluate)

    env_spec = acme.make_environment_spec(env)

    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = fruitbot_helpers.load_agent_settings(config['agent'], env_spec, config_kwargs=config)

  elif env_type == 'goto':
    env = borsa_helpers.make_environment(
      setting=setting,
      evaluation=evaluate)

    env_spec = acme.make_environment_spec(env)

    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = borsa_helpers.load_agent_settings(config['agent'], env_spec, config_kwargs=config)
  elif env_type == 'minihack':
    env = minihack_helpers.make_environment(
      setting=setting,
      evaluation=evaluate)

    env_spec = acme.make_environment_spec(env)

    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = minihack_helpers.load_agent_settings(config['agent'], env_spec, config_kwargs=config)
  else:
    raise RuntimeError(env_type)

  settings=dict(
      config=config,
      NetworkCls=NetworkCls,
      NetKwargs=NetKwargs,
      LossFn=LossFn,
      LossFnKwargs=LossFnKwargs,
      env_spec=env_spec,
    )

  return env, settings

def main(_):
  # basepath = './results/fruitbot/'
  # searches = dict(
  #     exploration1='msf_taskgen_easy-5/agen=exploration1,sett=taskgen_long_easy,valu=0.5,eval=train',
  #     uvfa='r2d1_taskgen_easy-5/*', #
  #   )

  all_searches=dict(
    msf2=dict(
        msf='ablate_shared-4/agen=exploration1,memo=None,modu=150,sepe=False,sepe=False',
        usfa='xl_respawn-grad-3/agen=usfa',
        uvfa='xl_respawn-grad-3/agen=r2d1',
        usfa_lstm='xl_respawn-grad-3/agen=usfa_lstm',
        uvfa_farm='xl_respawn-farm-4/agen=r2d1_farm*True*False',
      ),
    fruitbot=dict(
        msf='taskgen_final-3/agen=exploration1,sett=taskgen_long_easy',
        uvfa='taskgen_final-3/agen=r2d1,sett=taskgen_long_easy',
        usfa_lstm='taskgen_final-3/agen=usfa_lstm,sett=taskgen_long_easy',
        uvfa_farm='taskgen_final-3/agen=r2d1_farm,sett=taskgen_long_easy',
      ),
    minihack=dict(
        msf='large_final-5/agen=exploration1,sett=room_large,*64',
        usfa_lstm='large_final-5/agen=usfa_lstm,sett=room_large,*128',
        uvfa='large_final-uvfa-5/agen=r2d1,sett=room_large,*',
        uvfa_farm='large_final-uvfa-5/agen=r2d1_farm,sett=room_large,*',
      )
    )

  folder2env=dict(
    msf2='goto',
    fruitbot='fruitbot',
    minihack='minihack'
    )

  folder2fps=dict(
    msf2=3,
    fruitbot=30,
    minihack=3
    )

  folder = FLAGS.folder
  basepath = f'./results/{folder}/'
  for agent, agent_search in all_searches[folder].items():
    generate_data(
      agent_name=agent,
      basepath=basepath,
      agent_data_path=agent_search,
      total_episodes=FLAGS.num_episodes,
      frame_rate=folder2fps[folder],
      load_env_agent_settings=functools.partial(load_env_agent_settings,
        env_type=folder2env[folder]
        ),
      overwrite=FLAGS.overwrite,
      ckpt=FLAGS.ckpts,
      video_path=FLAGS.video_path)


if __name__ == '__main__':
  app.run(main)
