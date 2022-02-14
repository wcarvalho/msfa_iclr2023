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
  evaluation_epsilon: float = 0.
  num_epsilons: int = 256
  variable_update_period: int = 400 # how often to update actor

  # Learner options
  burn_in_length: int = 40  # burn in during learning
  trace_length: int = 40  # how long training should be
  sequence_period: int = 40  # how often to add
  learning_rate: float = 5e-5
  bootstrap_n: int = 5
  seed: int = 1
  max_number_of_steps: int = 10_000_000
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 4

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 0.0 # 0.0=single process
  min_replay_size: int = 1000
  max_replay_size: int = 200_000
  batch_size: int = 32
  store_lstm_state: bool = True
  prefetch_size: int = 0
  num_parallel_calls: int = 1
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9

  # Network hps
  memory_size = 512
  out_hidden_size = 128


@dataclasses.dataclass
class USFAConfig(R2D1Config):
  """Extra configuration options for USFA agent."""
  npolicies: int = 10 # number of policies to sample
  variance: float = 0.1
  # Network hps
  policy_size = 32
  batch_size: int = 20
  cumulant_hidden_size: int=128 # hidden size for cunmulant pred

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig):
  """Extra configuration options for USFA agent."""
  mixture: str='unique'
  cumtype: str='sum'



@dataclasses.dataclass
class FarmConfig:
  """Extra configuration options for FARM module."""

  # Network hps
  module_size: int = 128
  nmodules: int = 4
  out_layers: int = 2

@dataclasses.dataclass
class FarmModelConfig:
  """Extra configuration options for FARM module."""

  # Network hps
  extra_negatives: int = 10
  temperature: float = 0.01
  model_coeff: float = 1e-3
  out_layers: int = 2
  model_layers: int = 2
  batch_size: int = 16
  activation: str='relu'


@dataclasses.dataclass
class RewardConfig:
  """Extra configuration options for USFA agent."""
  reward_coeff: float = 0.1 # coefficient for loss
  reward_loss: str = 'l2' # type of regression. L2 vs. binary cross entropy


@dataclasses.dataclass
class VAEConfig:
  """Extra configuration options for USFA agent."""
  vae_coeff: float = 1e-4 # coefficient for loss
  latent_source: str = "samples" # coefficient for loss
  latent_dim: int = 128 # latent dim for compression
  beta: float = 25 # beta for KL

