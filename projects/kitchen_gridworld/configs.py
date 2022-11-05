"""Config."""
import dataclasses

from projects.msf.configs import *
from projects.msf import configs

@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  """Configuration options for R2D2 agent."""
  max_number_of_steps: int = 40_000_000

  # Network hps
  out_hidden_size: int = 512
  r2d1_loss: str = 'transformed_n_step_q_learning'
  embed_task_dim: int=16
  samples_per_insert: float = 6.0 # 0.0=infinite

@dataclasses.dataclass
class ModR2d1Config(R2D1Config):
  """Extra configuration options for USFA agent."""
  policy_size: int = 32 # embed dim for task input to q-fn
  policy_layers: int = 2 # number of layers to embed task for input to q-fn
  struct_w: bool = True # break up task per module
  dot_qheads: bool = True # break up q-heads and dot-product
  nmodules: int = 4 # break up q-heads and dot-product
  module_task_dim: int=0 # task dim per module. if 0, use embed_task_dim and divide by nmodules

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  """Extra configuration options for USFA agent."""
  policy_layers: int = 2
  eval_task_support: str='eval'

@dataclasses.dataclass
class RewardConfig(configs.RewardConfig):
  """Extra configuration options for USFA agent."""
  reward_coeff: float = 10.0 # coefficient for reward loss
  value_coeff: float = 0.5 # coefficient for value loss

@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  """Extra configuration options for FARM module."""

  # Network hps
  memory_size: int = 512
  module_size: int = None
  nmodules: int = 4
  module_attn_heads: float = .5  # how many attention heads between modules

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  """Extra configuration options for USFA agent."""
  struct_policy_input: bool = True # break up task per module

  module_task_dim: int=1 # task dim per module. if 0, use embed_task_dim and divide by nmodules

@dataclasses.dataclass
class FarmModelConfig(FarmConfig):
  """Extra configuration options for FARM module."""

  # Network hps
  temperature: float = 0.01
  reward_coeff: float = 10.0 # coefficient for reward loss
  cumulant_const: str='concat'  # whether to use delta between states as cumulant
  out_layers: int = 0
  model_layers: int = 2
  activation: str='relu'
  seperate_model_params: bool=True # seperate parameters per transition fn
  normalize_step: bool=False # whether to normalize delta step in TimeContrastLoss
  contrast_module_coeff: float = 0.0
  contrast_module_pred: str = 'delta'
  contrast_time_coeff: float = 0.0
  extra_module_negatives: int = 4
  extra_time_negatives: int = 0



@dataclasses.dataclass
class LangConfig:
  max_vocab_size: int = 35
  task_embedding: str='language'
  word_dim: int = 128  # dimension of word and sentence embeddings
  word_initializer: str = 'RandomNormal'
  word_compress: str = 'last'
  embed_task_dim: int = 16  # dimension of task
  lang_activation: str = 'none'  # whether to apply tanh
  bag_of_words: bool=False