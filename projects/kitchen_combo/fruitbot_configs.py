from projects.msf.configs import *
from projects.msf import configs


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  vision_torso: str = 'impala'
  memory_size: int = 256
  out_hidden_size: int = 512
  out_q_layers: int = 2
  r2d1_loss: str = 'n_step_q_learning'

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  reward_coeff: float = 10
  value_coeff: float = 0.5
  eval_task_support: str='train'

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  reward_coeff: float = 10
  value_coeff: float = 0.5

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  nmodules: int = 2
  module_attn_heads: int = 1
  module_task_dim: int = 0 # divide embed_task_dim by nmodules
