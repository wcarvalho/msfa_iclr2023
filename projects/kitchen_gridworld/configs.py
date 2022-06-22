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
  task_gate: str='none'
  task_embedding: str='language'
  embed_task_dim: int=16
  samples_per_insert: float = 0.0 # 0.0=infinite

@dataclasses.dataclass
class ModR2d1Config(R2D1Config):
  """Extra configuration options for USFA agent."""
  policy_size: int = 32 # embed dim for task input to q-fn
  policy_layers: int = 2 # number of layers to embed task for input to q-fn
  struct_w: bool = False # break up task per module
  dot_qheads: bool = False # break up q-heads and dot-product
  module_task_dim: int=0 # task dim per module. if 0, use embed_task_dim and divide by nmodules

@dataclasses.dataclass
class USFAConfig(R2D1Config, configs.USFAConfig):
  """Extra configuration options for USFA agent."""
  policy_layers: int = 2

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
class ModularUSFAConfig(USFAConfig):
  """Extra configuration options for USFA agent."""
  normalize_delta: bool = True # whether to normalize delta between states
  normalize_state: bool = True # whether to normalize delta between states
  embed_position: int = 0 # whether to add position embeddings to modules
  position_hidden: bool = False # whether to add position embeddings to modules
  struct_policy_input: bool = False # break up task per module

  cumulant_source: str = 'lstm' # whether to normalize cumulants
  phi_conv_size: int = 0 # size of conv for cumulants
  seperate_cumulant_params: bool=True # seperate parameters per cumulant set
  seperate_value_params: bool=False # seperate parameters per SF set
  phi_l1_coeff: float = 0.00 # coefficient for L1 on phi
  w_l1_coeff: float = 0.00 # coefficient for L1 on w
  module_l1: bool = False # apply L1 per module or for all phi
  cov_loss: str = 'l1_cov' # apply L1 per module or for all phi

  sf_net: str = 'independent'
  sf_net_heads: int = 2
  sf_net_layers: int=1
  sf_net_attn_size: int = 256

  phi_net: str = 'independent'
  phi_net_heads: int = 2
  phi_net_layers: int=1

  relate_w_init: float=2.
  resid_w_init: float=2.
  relate_b_init: float=2.
  resid_mlp: bool=False
  relate_residual: str="sigtanh"
  res_relu_gate: bool=True
  layernorm_rel: bool=False

  task_gate: str='none'
  module_task_dim: int=1 # task dim per module. if 0, use embed_task_dim and divide by nmodules
  qaux_mask_loss: bool=True
  sf_mask_loss: bool=True
  phi_mask_loss: bool=True


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
  word_dim: int = 128  # dimension of word and sentence embeddings
  word_initializer: str = 'RandomNormal'
  word_compress: str = 'last'
  embed_task_dim: int = 16  # dimension of task
  lang_activation: str = 'none'  # whether to apply tanh
