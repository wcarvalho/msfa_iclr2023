from projects.kitchen_gridworld.configs import *
from projects.kitchen_gridworld import configs


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  vision_torso: str = 'impala'
  memory_size: int = 256

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  reward_coeff: float = 10
  value_coeff: float = 0.5

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  reward_coeff: float = 10
  value_coeff: float = 0.5

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  nmodules: int = 4
  module_attn_heads: int = 2
  module_task_dim: int = 0 # divide embed_task_dim by nmodules
