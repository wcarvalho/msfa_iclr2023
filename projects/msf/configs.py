"""Config."""
import dataclasses

from acme.adders import reverb as adders_reverb
from agents.td_agent import configs
import rlax


@dataclasses.dataclass
class R2D1Config(configs.R2D1Config):
  """Configuration options for R2D2 agent."""
  discount: float = 0.99
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.01
  num_epsilons: int = 256
  variable_update_period: int = 400 # how often to update actor

  # Learner options
  burn_in_length: int = 0  # burn in during learning
  trace_length: int = 40  # how long training should be
  sequence_period: int = 40  # how often to add
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  step_penalty: float = 0.0
  seed: int = 3
  max_number_of_steps: int = 3_000_000
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.
  loss_coeff: float = 1.0

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 4

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 6.0 # 0.0=single process
  min_replay_size: int = 100
  max_replay_size: int = 100_000
  batch_size: int = 32
  store_lstm_state: bool = True
  prefetch_size: int = 0
  num_parallel_calls: int = 1
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.0
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9


  # Network hps
  memory_size: int = 512
  out_hidden_size: int = 128
  out_q_layers: int = 1
  task_embedding: str='none'
  eval_network: bool = True
  vision_torso: str = 'atari'
  r2d1_loss: str = 'n_step_q_learning'

@dataclasses.dataclass
class NoiseConfig(R2D1Config):
  """Extra configuration options for R2D1 + noise agent."""
  variance: float = 0.5


@dataclasses.dataclass
class USFAConfig(R2D1Config):
  """Extra configuration options for USFA agent."""
  npolicies: int = 10 # number of policies to sample
  variance: float = 0.5
  # Network hps
  policy_size: int = 32
  policy_layers: int = 0
  batch_size: int = 32
  cumulant_hidden_size: int=256 # hidden size for cumulant pred
  cumulant_layers: int=2 # hidden size for cumulant pred
  embed_task: bool = False  # whether to embed task
  normalize_task: bool = False # whether to normalize task embedding
  eval_network: bool = True
  duelling: bool = False
  z_as_train_task: bool = False  # whether to dot-product SF with z-vector (True) or w-vector (False)
  state_hidden_size: int = 0
  multihead: bool = False
  concat_w: bool = False
  sf_loss: str = 'n_step_q_learning' # whether to use q_lambda or n-step q-learning for objective
  lambda_: float = .9 # lambda for q-lambda
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR


  phi_l1_coeff: float = 0.0 # coefficient for L1 on phi
  w_l1_coeff: float = 0.0 # coefficient for L1 on w
  cov_coeff: float = None # coeff for covariance loss on phi
  sf_layernorm: str = 'none' # coefficient for L1 on phi
  task_gate: str='none'
  sf_mask_loss: bool=True
  phi_mask_loss: bool=True
  eval_task_support: str='train' # include eval task in support
  stop_w_grad: bool=False
  stop_z_grad: bool=False
  target_phi: bool=False

@dataclasses.dataclass
class QAuxConfig:
  """Extra configuration options when doing QAux loss over SF."""
  loss_coeff: float = 1.0
  q_aux_anneal: int = 0.0
  q_aux_end_val: float = 0.0
  qaux_mask_loss: bool=True



@dataclasses.dataclass
class RewardConfig:
  """Extra configuration options for USFA agent."""
  reward_coeff: float = 1 # coefficient for reward loss
  value_coeff: float = 0.5 # coefficient for value loss
  reward_loss: str = 'l2' # type of regression. L2 vs. binary cross entropy
  balance_reward: float = .25 # whether to balance dataset and what percent of nonzero to keep
  q_aux: str="single"
  normalize_cumulants: bool = False # whether to normalize cumulants
  cumulant_act: str = 'identity' # activation on cumulants
  cumulant_const: str='concat'  # whether to use delta between states as cumulant

@dataclasses.dataclass
class FarmConfig:
  """Extra configuration options for FARM module."""

  # Network hps
  memory_size: int = None
  module_size: int = 128
  nmodules: int = 4
  out_layers: int = 0
  module_attn_size: int = None
  module_attn_heads: int = 2  # how many attention heads between modules
  shared_module_attn: bool = True # share params for module attention
  projection_dim: int = 16
  farm_vmap: str = "lift"  # vmap over different parameter sets 
  image_attn: bool = True # whether to use feature attention on image
  farm_task_input: bool = False # give task as input to FARM
  farm_policy_task_input: bool = False # give task as input to FARM policy

  recurrent_conv: bool = False # whether to use feature attention on image
  normalize_attn: bool = False # whether to use feature attention on image


@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig):
  """Extra configuration options for USFA agent."""
  normalize_delta: bool = True # whether to normalize delta between states
  normalize_state: bool = True # whether to normalize delta between states
  embed_position: int = 0 # whether to add position embeddings to modules
  position_hidden: bool = False # whether to add position embeddings to modules

  seperate_cumulant_params: bool=True # seperate parameters per cumulant set
  seperate_value_params: bool=False # seperate parameters per SF set
  struct_policy_input: bool=False

  sf_net: str = 'independent'
  sf_net_heads: int = 2
  sf_net_layers: int=1
  sf_net_attn_size: int = 256

  phi_net: str = 'independent'
  phi_net_heads: int = 2
  phi_net_layers: int=1
  cumulant_const: str='concat'  # whether to use delta between states as cumulant

  relate_w_init: float=2.
  resid_w_init: float=2.
  relate_b_init: float=2.
  resid_mlp: bool=False
  relate_residual: str="sigtanh"
  res_relu_gate: bool=True
  layernorm_rel: bool=False
  argmax_mod: bool=False # argmax over modules
  qaux_mask_loss: bool=True
  sf_mask_loss: bool=True

@dataclasses.dataclass
class FarmModelConfig(FarmConfig):
  """Extra configuration options for FARM module."""

  # Network hps
  temperature: float = 0.01
  reward_coeff: float = 1 # coefficient for reward loss
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
