from projects.kitchen_gridworld import configs
from projects.kitchen_gridworld.configs import *


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  task_embedding: str='none'
  embed_task_dim: int=6
  out_hidden_size: int=512

@dataclasses.dataclass
class USFAConfig(configs.USFAConfig, R2D1Config):
  reward_coeff: float = 10

@dataclasses.dataclass
class ModularUSFAConfig(configs.ModularUSFAConfig, USFAConfig):
  reward_coeff: float = 10

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  nmodules: int = 3
  memory_size: int = None
  module_attn_heads: int = 1
