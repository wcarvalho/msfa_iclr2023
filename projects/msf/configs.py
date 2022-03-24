"""Config."""
import dataclasses

from acme.adders import reverb as adders_reverb
from acme.agents.jax.r2d2 import config as r2d2_config
import rlax


@dataclasses.dataclass
class R2D1Config:
  """Configuration options for R2D2 agent."""
  discount: float = 0.99
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.0
  num_epsilons: int = 256
  variable_update_period: int = 400 # how often to update actor

  # Learner options
  burn_in_length: int = 0  # burn in during learning
  trace_length: int = 40  # how long training should be
  sequence_period: int = 40  # how often to add
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  seed: int = 1
  max_number_of_steps: int = 2_000_000
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.
  loss_coeff: float = 1.0

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 4

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 0.0 # 0.0=single process
  min_replay_size: int = 10_000
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
  memory_size = 512
  out_hidden_size = 128
  eval_network: bool = True
  vision_torso: str = 'atari'

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
  policy_size = 32
  policy_layers = 0
  batch_size: int = 32
  cumulant_hidden_size: int=128 # hidden size for cumulant pred
  embed_task: bool = False  # whether to embed task
  normalize_task: bool = False # whether to normalize task embedding
  eval_network: bool = True
  duelling: bool = False
  z_as_train_task: bool = False  # whether to dot-product SF with z-vector (True) or w-vector (False)
  state_hidden_size: int = 0
  multihead: bool = False
  concat_w: bool = False
  sf_loss: str = 'n_step_q_learning_regular' # whether to use q_lambda or n-step q-learning for objective
  lambda_: float = .9 # lambda for q-lambda

@dataclasses.dataclass
class QAuxConfig:
  """Extra configuration options when doing QAux loss over SF."""
  loss_coeff: float = 1e-2


@dataclasses.dataclass
class RewardConfig:
  """Extra configuration options for USFA agent."""
  reward_coeff: float = 1e-3 # coefficient for reward loss
  value_coeff: float = 1. # coefficient for value loss
  reward_loss: str = 'l2' # type of regression. L2 vs. binary cross entropy
  balance_reward: float = .25 # whether to balance dataset and what percent of nonzero to keep
  q_aux: str="ensemble"
  normalize_cumulants: bool = False # whether to normalize cumulants
  cumulant_act: str = 'identity' # activation on cumulants
  cumulant_const: str='concat'  # whether to use delta between states as cumulant

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig):
  """Extra configuration options for USFA agent."""
  mixture: str='unique'  # how to mix FARM modules
  aggregation: str='concat'  # how to aggregate modules for cumulant
  normalize_delta: bool = True # whether to normalize delta between states



@dataclasses.dataclass
class FarmConfig:
  """Extra configuration options for FARM module."""

  # Network hps
  module_size: int = 128
  nmodules: int = 4
  out_layers: int = 0
  shared_attn_params: bool = False
  module_attn_size: int = 64
  module_attn_heads: int = 0  # how many attention heads between modules
  shared_module_attn: bool = False
  projection_dim: int = 16
  farm_vmap: bool = False

@dataclasses.dataclass
class FarmModelConfig(FarmConfig):
  """Extra configuration options for FARM module."""

  # Network hps
  extra_negatives: int = 10
  temperature: float = 0.01
  model_coeff: float = 1
  out_layers: int = 2
  model_layers: int = 2
  batch_size: int = 16
  activation: str='relu'




@dataclasses.dataclass
class VAEConfig:
  """Extra configuration options for USFA agent."""
  vae_coeff: float = 1e-4 # coefficient for loss
  latent_source: str = "samples" # coefficient for loss
  latent_dim: int = 128 # latent dim for compression
  beta: float = 25 # beta for KL

