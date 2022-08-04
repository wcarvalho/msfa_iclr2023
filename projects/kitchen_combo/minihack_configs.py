from projects.msf.configs import *
from projects.msf import configs


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  vision_torso: str = 'impala'
  memory_size: int = 256
  out_hidden_size: int = 512
  embed_task_dim: int = 12
  out_q_layers: int = 2
  r2d1_loss: str = 'transformed_n_step_q_learning'
  max_number_of_steps: int = 10_000_000
  max_replay_size: int = 70_000
  # min_replay_size: int = 10_000
  samples_per_insert: 4.0 # resize is expensive, so lower is better.
  importance_sampling_exponent: float = 0.6 
  batch_size: int = 32

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  reward_coeff: float = 10.0
  value_coeff: float = 0.5
  memory_size: int = 300

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  reward_coeff: float = 10.0
  value_coeff: float = 0.5
  sf_share_output: bool=False

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  module_size: int = None
  nmodules: int = 3
  memory_size: int = 240 # based on fruitbot
  module_attn_heads: float = .5
  module_task_dim: int = 0 # divide embed_task_dim by nmodules
