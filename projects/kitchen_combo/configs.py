from projects.kitchen_gridworld.configs import *
from projects.kitchen_gridworld import configs


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  task_embedding: str='none'
  embed_task_dim: int=6

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  reward_coeff: float = 10

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  reward_coeff: float = 10

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  nmodules: int = 3
  module_size: int = 128
  memory_size: int = None
  module_attn_heads: int = 1
