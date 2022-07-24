from projects.kitchen_gridworld.configs import *
from projects.kitchen_gridworld import configs


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config): # 1.5M
  task_embedding: str='none'
  embed_task_dim: int=16
  out_q_layers: int = 2
  out_hidden_size: int = 512
  word_dim: int=128
  samples_per_insert: float = 6.0


@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig): # 1.6M
  reward_coeff: float = 1
  value_coeff: float = 0.5

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig): # 1.4M
  reward_coeff: float = 1
  value_coeff: float = 0.5

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  module_size: int = None
  nmodules: int = None
  module_attn_heads: float = .5
  module_task_dim: int = 1 # divide embed_task_dim by nmodules
