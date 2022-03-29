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
  burn_in_length: int = 0  # burn in during learning
  trace_length: int = 30  # how long training should be
  sequence_period: int = 40  # how often to add
  learning_rate: float = 5e-5
  bootstrap_n: int = 5
  seed: int = 1
  max_number_of_steps: int = 10_000_000
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
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9

  # Network hps
  memory_size = 512
  out_hidden_size = 128
  eval_network: bool = True
  # max_vocab_size: int = 30
  # word_dim: int = 128  # dimension of both word and task (sentence) embeddings


@dataclasses.dataclass
class USFAConfig(R2D1Config):
  """Extra configuration options for USFA agent."""
  npolicies: int = 10 # number of policies to sample
  variance: float = 0.5
  # Network hps
  policy_size = 32
  policy_layers = 0
  batch_size: int = 20
  cumulant_hidden_size: int=128 # hidden size for cumulant pred
  embed_task: bool = False  # whether to embed task
  normalize_task: bool = False # whether to normalize task embedding
  normalize_cumulants: bool = True # whether to normalize task embedding
  balance_reward: bool = False # whether to normalize task embedding
  eval_network: bool = True
  duelling: bool = False
  z_as_train_task: bool = True
  state_hidden_size: int = 0
