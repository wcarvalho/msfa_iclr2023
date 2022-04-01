"""Config."""
import dataclasses

from acme.adders import reverb as adders_reverb
from acme.agents.jax.r2d2 import config as r2d2_config
import rlax


@dataclasses.dataclass
class R2D1Config(r2d2_config.R2D2Config):
  """Configuration options for R2D2 agent."""
  discount: float = 0.997
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.
  num_epsilons: int = 256
  variable_update_period: int = 400 # how often to update actor

  # Learner options
  burn_in_length: int = 40  # burn in during learning
  trace_length: int = 80  # how long training should be
  sequence_period: int = 40  # how often to add
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  seed: int = 1
  max_number_of_steps: int = 10_000_000
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 1

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 0.0 # 0.0=single process
  min_replay_size: int = 50_000
  max_replay_size: int = 500_000
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
class USFAConfig(R2D1Config):
  """Extra configuration options for USFA agent."""
  npolicies: int = 10 # number of policies to sample
  variance: float = 0.1

@dataclasses.dataclass
class USFARewardConfig(USFAConfig):
  """Extra configuration options for USFA agent."""
  reward_coeff: float = 0.01 # coefficient for loss
  reward_loss: str = 'l2' # type of regression. L2 vs. binary cross entropy

