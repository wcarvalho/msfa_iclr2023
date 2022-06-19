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
  evaluation_epsilon: float = 0.0
  num_epsilons: int = 256
  variable_update_period: int = 400 # how often to update actor

  # Learner options
  burn_in_length: int = 0  # burn in during learning
  trace_length: int = 40  # how long training should be
  sequence_period: int = 40  # how often to add
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  seed: int = 3
  max_number_of_steps: int = 10_000_000
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
  max_gradient_norm: float = 80.0  # For gradient clipping.
  loss_coeff: float = 1.0
  schedule_end: int = None
  final_lr_scale: float = 1e-5

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 4

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 6.0 # 0.0=single process
  min_replay_size: int = 10_000
  #min_replay_size = 100
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
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
  memory_size: int = 512
  out_hidden_size: int = 512
=======
  memory_size = 512
  out_hidden_size = 128
>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py
  eval_network: bool = True
  vision_torso: str = 'atari'
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
  r2d1_loss: str = 'transformed_n_step_q_learning'
  task_gate: str='none'
  task_embedding: str='language'
  embed_task_dim: int=16
=======
  r2d1_loss: str = 'n_step_q_learning'

>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py

@dataclasses.dataclass
class NoiseConfig(R2D1Config):
    """Extra configuration options for R2D1 + noise agent."""
    variance: float = 0.5

@dataclasses.dataclass
class ModR2d1Config(R2D1Config):
  """Extra configuration options for USFA agent."""
  policy_size: int = 32 # embed dim for task input to q-fn
  policy_layers: int = 2 # number of layers to embed task for input to q-fn
  struct_w: bool = False # break up task per module
  dot_qheads: bool = False # break up q-heads and dot-product
  module_task_dim: int=0 # task dim per module. if 0, use embed_task_dim and divide by nmodules

@dataclasses.dataclass
class USFAConfig(R2D1Config):
  """Extra configuration options for USFA agent."""
  npolicies: int = 5 # number of policies to sample
  variance: float = 0.5
  # Network hps
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
  policy_size: int = 32
  policy_layers: int = 2 # [DIFF FROM MSF]
=======
  policy_size = 32
  policy_layers = 0
>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py
  batch_size: int = 32
  cumulant_hidden_size: int=128 # hidden size for cumulant pred
  cumulant_dimension: int = 9 # actual cumulant dimensions NOT USED RIGHT NOW BECAUSE NOT TASK EMBEDDING
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
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
  phi_l1_coeff: float = 0.0 # coefficient for L1 on phi
  w_l1_coeff: float = 0.0 # coefficient for L1 on w
  cov_coeff: float = None # coeff for covariance loss on phi
  sf_layernorm: str = 'none' # coefficient for L1 on phi
  task_gate: str='none'
  sf_mask_loss: bool=False
  eval_task_support: str='train' # include eval task in support
=======

>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py

@dataclasses.dataclass
class QAuxConfig:
  """Extra configuration options when doing QAux loss over SF."""
  loss_coeff: float = 1.0
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
  q_aux_anneal: int = 0.0
  q_aux_end_val: float = 0.0
  qaux_mask_loss: bool=False
  stop_w_grad: bool=True
=======
  q_aux_anneal: int = 0
  q_aux_end_val: float = 1e-2
>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py


@dataclasses.dataclass
class RewardConfig:
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
  """Extra configuration options for USFA agent."""
  reward_coeff: float = 10.0 # coefficient for reward loss
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
  memory_size: int = 512
  module_size: int = None
  nmodules: int = 4
  out_layers: int = 0
  module_attn_size: int = None
  module_attn_heads: float = .5  # how many attention heads between modules
  shared_module_attn: bool = True # share params for module attention
  projection_dim: int = 16
  farm_vmap: str = "lift"  # vmap over different parameter sets 
  image_attn: bool = True # whether to use feature attention on image
  recurrent_conv: bool = False # whether to use feature attention on image
  normalize_attn: bool = False # whether to use feature attention on image
  farm_task_input: bool = False # give task as input to FARM
  farm_policy_task_input: bool = False # give task as input to FARM policy
=======
    """Extra configuration options for USFA agent."""
    reward_coeff: float = 1e-3  # coefficient for reward loss
    value_coeff: float = 1.  # coefficient for value loss
    reward_loss: str = 'l2'  # type of regression. L2 vs. binary cross entropy
    balance_reward: float = .25  # whether to balance dataset and what percent of nonzero to keep
    q_aux: str = "single"
    normalize_cumulants: bool = False  # whether to normalize cumulants
    cumulant_act: str = 'identity'  # activation on cumulants
    cumulant_const: str = 'concat'  # whether to use delta between states as cumulant
>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py


@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig):
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
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
  module_task_dim: int=0 # task dim per module. if 0, use embed_task_dim and divide by nmodules
  qaux_mask_loss: bool=True
  sf_mask_loss: bool=True
=======
    """Extra configuration options for USFA agent."""
    mixture: str = 'unique'  # how to mix FARM modules
    aggregation: str = 'concat'  # how to aggregate modules for cumulant
    normalize_delta: bool = True  # whether to normalize delta between states
>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py


@dataclasses.dataclass
class FarmConfig:
    """Extra configuration options for FARM module."""

    # Network hps
    module_size: int = 128
    nmodules: int = 4
    out_layers: int = 0
    module_attn_size: int = 64
    module_attn_heads: int = 0  # how many attention heads between modules
    shared_module_attn: bool = False  # share params for module attention
    projection_dim: int = 16
    farm_vmap: str = "lift"  # vmap over different parameter sets
    image_attn: bool = True  # whether to use feature attention on image
    farm_task_input: bool = True  # give task as input to FARM
    farm_policy_task_input: bool = False  # give task as input to FARM policy
    seperate_cumulant_params: bool = True  # seperate parameters per cumulant set
    seperate_value_params: bool = True  # seperate parameters per SF set


@dataclasses.dataclass
class FarmModelConfig(FarmConfig):
    """Extra configuration options for FARM module."""

    # Network hps
    extra_negatives: int = 4
    temperature: float = 0.01
    model_coeff: float = .1
    reward_coeff: float = 1e-4  # coefficient for reward loss
    out_layers: int = 0
    model_layers: int = 2
    activation: str = 'relu'
    cumulant_const: str = 'delta'  # whether to use delta between states as cumulant
    seperate_model_params: bool = True  # seperate parameters per transition fn


@dataclasses.dataclass
<<<<<<< HEAD:projects/kitchen_gridworld/configs.py
class LangConfig:
  max_vocab_size: int = 35
  word_dim: int = 128  # dimension of word and sentence embeddings
  word_initializer: str = 'RandomNormal'
  word_compress: str = 'last'
  embed_task_dim: int = 16  # dimension of task
  lang_tanh: bool = False  # whether to apply tanh
=======
class VAEConfig:
    """Extra configuration options for USFA agent."""
    vae_coeff: float = 1e-4  # coefficient for loss
    latent_source: str = "samples"  # coefficient for loss
    latent_dim: int = 128  # latent dim for compression
    beta: float = 25  # beta for KL
>>>>>>> parent of d34fcbe (maybe we can merge this?):projects/colocation/configs.py

