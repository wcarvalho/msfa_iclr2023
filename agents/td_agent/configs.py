"""Config."""
import dataclasses

from acme.adders import reverb as adders_reverb
from acme.agents.jax.r2d2 import config as r2d2_config
import rlax


@dataclasses.dataclass
class R2D1Config:
  """Configuration options for R2D2 agent."""
  discount: float = 0.997
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.
  num_epsilons: int = 256
  variable_update_period: int = 400

  # Learner options
  burn_in_length: int = 40
  trace_length: int = 80
  sequence_period: int = 40
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 1

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 4.0
  min_replay_size: int = 50_000
  max_replay_size: int = 100_000
  batch_size: int = 64
  prefetch_size: int = 2
  store_lstm_state: bool = True
  num_parallel_calls: int = 16
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9


@dataclasses.dataclass
class USFAConfig:
  """Configuration options for USFA agent."""
  discount: float = 0.997
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.
  num_epsilons: int = 256
  variable_update_period: int = 400
  npolicies: int = 10 # number of policies to sample

  # Learner options
  burn_in_length: int = 40
  trace_length: int = 80
  sequence_period: int = 40
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 1

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 4.0
  min_replay_size: int = 50_000
  max_replay_size: int = 100_000
  batch_size: int = 64
  prefetch_size: int = 2
  store_lstm_state: bool = True
  num_parallel_calls: int = 16
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9
