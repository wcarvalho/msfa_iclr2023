from projects.msf.configs import *
from projects.msf import configs


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  vision_torso: str = 'impala'
  memory_size: int = 256
  out_hidden_size: int = 512
  out_q_layers: int = 2
  r2d1_loss: str = 'transformed_n_step_q_learning'
  max_number_of_steps: int = 10_000_000
  max_replay_size: int = 80_000
  min_replay_size: int = 10_000
  importance_sampling_exponent: float = 0.6 # for more fair generalization comparison
  batch_size: int = 32

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  reward_coeff: float = 10
  value_coeff: float = 0.5
  memory_size: int = 300
  eval_task_support: str=None

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  reward_coeff: float = 1 # > 1.0 fails in this domain 
  value_coeff: float = 0.5
  eval_task_support: str=None

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  module_size: int = 80
  nmodules: int = None
  module_attn_heads: int = 1
  module_task_dim: int = 1 # divide embed_task_dim by nmodules
