"""
"""
import os
from absl import app
from absl import flags
import acme


FLAGS = flags.FLAGS

from projects.common.create_analysis_data import generate_data
from projects.kitchen_combo import fruitbot_helpers

def load_env_agent_settings(config : dict, setting=None, evaluate=True):
  setting = setting or config['setting']

  env = fruitbot_helpers.make_environment(
    setting=setting,
    train_in_eval=False,
    evaluation=evaluate)

  env_spec = acme.make_environment_spec(env)

  config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, _, _ = fruitbot_helpers.load_agent_settings(config['agent'], env_spec, config_kwargs=config)

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
  basepath = './results/fruitbot/'
  searches = dict(
      msf='msf_taskgen_easy-5/agen=msf,sett=taskgen_long_easy,valu=0.5,eval=train',
      uvfa='r2d1_taskgen_easy-5/*', #
    )

  for agent, agent_search in searches.items():
    generate_data(
      agent_name=agent,
      basepath=basepath,
      agent_data_path=agent_search,
      load_env_agent_settings=load_env_agent_settings,
      overwrite=FLAGS.overwrite,
      ckpt=FLAGS.ckpts,
      video_path=FLAGS.video_path)


if __name__ == '__main__':
  app.run(main)
