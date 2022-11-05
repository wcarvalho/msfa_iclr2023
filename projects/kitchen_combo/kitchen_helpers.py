import os.path
import yaml

import acme
import functools

from acme import wrappers
import dm_env
import rlax


from utils import ObservationRemapWrapper
from utils import data as data_utils

from agents import td_agent
from agents.td_agent import losses

from losses.contrastive_model import ModuleContrastLoss, TimeContrastLoss
from losses import cumulants
from losses import msfa_stats


# from projects.msf.helpers import q_aux_sf_loss
from projects.kitchen_combo import kitchen_configs as configs
from projects.common_usfm import agent_loading
from projects.common_usfm import nets

from envs.acme.tasks_wrapper import TrainTasksWrapper
from envs.acme.multitask_kitchen import MultitaskKitchen
from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, MissionIntegerWrapper
from envs.babyai_kitchen.utils import InstructionsPreprocessor



# ======================================================
# Environment
# ======================================================
def make_environment(evaluation: bool = False,
                     tile_size=8,
                     room_size=6,
                     num_dists=0,
                     step_penalty=0.0,
                     task_reps='lesslang',
                     max_text_length=10,
                     symbolic=False,
                     path='.',
                     setting='',
                     struct_and=False,
                     task_reset_behavior='none',
                     debug=False,
                     nseeds=500,
                     **kwargs,
                     ) -> dm_env.Environment:
  setting = setting or 'gen_long_seeds'
  """Loads environments."""

  task_reps_file = f"envs/babyai_kitchen/tasks/task_reps/{task_reps}.yaml"
  task_reps_file = os.path.join(path, task_reps_file)
  assert os.path.exists(task_reps_file)
  with open(task_reps_file, 'r') as f:
    task_reps = yaml.load(f, Loader=yaml.SafeLoader)

  tasks_file = f"envs/babyai_kitchen/tasks/v1/{setting}.yaml"
  tasks_file = os.path.join(path, tasks_file)
  assert os.path.exists(tasks_file)
  
  with open(tasks_file, 'r') as f:
    tasks = yaml.load(f, Loader=yaml.SafeLoader)

  if evaluation and 'test' in tasks:
    task_dicts = tasks['test']
    train_task_dicts = tasks['train']
    env_kwargs=dict(task_reset_behavior=task_reset_behavior)
  else:
    task_dicts = tasks['train']
    train_task_dicts = tasks['train']
    env_kwargs=dict()

  if 'task_reps' in tasks:
    task_reps = tasks['task_reps']

  instr_preproc = InstructionsPreprocessor(
    path=os.path.join(path, "data/babyai_kitchen/vocab.json"))

  env_wrappers = [functools.partial(MissionIntegerWrapper,
        instr_preproc=instr_preproc,
        max_length=max_text_length,
        struct_and=struct_and and evaluation)]

  if not symbolic:
    env_wrappers.append(
      functools.partial(RGBImgPartialObsWrapper, tile_size=tile_size))

  nseeds=0 if evaluation else nseeds
  env = MultitaskKitchen(
    task_dicts=task_dicts,
    tasks_file=tasks,
    tile_size=tile_size,
    path=path,
    num_dists=num_dists,
    task_reps=task_reps,
    step_penalty=step_penalty,
    room_size=room_size,
    wrappers=env_wrappers,
    symbolic=symbolic,
    debug=debug,
    nseeds=nseeds,
    **env_kwargs,
    **kwargs
    )

  # wrappers for dm_env: used by agent/replay buffer
  wrapper_list = [
    functools.partial(ObservationRemapWrapper,
        remap=dict(mission='task'))]

  wrapper_list.append(
    functools.partial(TrainTasksWrapper,
        instr_preproc=instr_preproc,
        max_length=max_text_length,
        task_reps=task_reps,
        train_tasks=[t['task_kinds'] for t in train_task_dicts],
      ),
    )

  wrapper_list += [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]



  return wrappers.wrap_all(env, wrapper_list)



# ======================================================
# Building Agent Networks
# ======================================================

def load_agent_settings(agent, env_spec, config_kwargs=None, env_kwargs=None, max_vocab_size=50):

  default_config = dict(max_vocab_size=max_vocab_size)
  default_config.update(config_kwargs or {})

  if agent == "r2d1":
  # Recurrent DQN (2.2M params)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[
        configs.R2D1Config(),
        configs.LangConfig(),
      ],
      task_input='qfn_concat',
      )

  elif agent == "r2d1_dot":
  # Recurrent DQN (2.2M params)
    default_config['memory_size'] = 550
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
    env_spec=env_spec,
    default_config=default_config,
    dataclass_configs=[
      configs.R2D1Config(),
      configs.LangConfig(),
    ],
    task_input='qfn_dot',
    )

  elif agent == "r2d1_no_task": 
  # UVFA + noise added to goal embedding
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      dataclass_configs=[configs.R2D1Config()],
      task_input='none',
      task_embedding='none',
      )

  elif agent == "r2d1_farm":
  # UVFA + FARM
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      NetworkCls = nets.r2d1_farm,
      dataclass_configs=[
        configs.R2D1Config(),
        configs.FarmConfig(),
        configs.LangConfig(),
      ],
    )


  elif agent == "r2d1_noise":
  # UVFA + FARM
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.r2d1(
      env_spec=env_spec,
      default_config=default_config,
      NetworkCls=nets.r2d1_noise,
      dataclass_configs=[
        configs.NoiseConfig(),
        configs.R2D1Config(),
        configs.LangConfig(),
      ],
    )
  elif agent == "modr2d1":
  # Recurrent DQN (2.2M params)
    raise NotImplementedError("Not implemented with common nets")
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.ModR2d1Config(),
        configs.FarmConfig(),
        configs.LangConfig(),
      ],
      dict_configs=default_config)

    NetworkCls=nets.modr2d1 # default: 2M params
    NetKwargs=dict(
      config=config,
      env_spec=env_spec,
      )
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      loss=config.r2d1_loss,
      mask_loss=config.q_mask_loss)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "usfa_lstm":
  # USFA + cumulants from LSTM + Q-learning (2.5M params)
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.usfa_lstm(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          configs.QAuxConfig(),
          configs.RewardConfig(),
          configs.USFAConfig(),
          configs.LangConfig(),
          ],
      )

  elif agent == "msf":
  # USFA + cumulants from FARM + Q-learning
    config, NetworkCls, NetKwargs, LossFn, LossFnKwargs = agent_loading.msf(
        env_spec=env_spec,
        default_config=default_config,
        dataclass_configs=[
          configs.QAuxConfig(),
          configs.RewardConfig(),
          configs.FarmConfig(),
          configs.ModularUSFAConfig(),
          configs.LangConfig(),
        ],
      )

  else:
    raise NotImplementedError(agent)

  loss_label=None
  eval_network=False
  return config, NetworkCls, NetKwargs, LossFn, LossFnKwargs, loss_label, eval_network