"""Implementation of Recurrent TD agents. Primarily for GVF based methods (e.g. USFA)."""

# from acme.agents.jax.r2d2.agents import DistributedRecurrentTDAgent
# from acme.agents.jax.r2d2.agents import DistributedR2D2FromConfig
from agents.td_agent.agents import TDAgent
from agents.td_agent.builder import TDBuilder
from agents.td_agent.configs import R2D1Config, USFAConfig
from agents.td_agent.losses import R2D2Learning, r2d2_loss_kwargs
from agents.td_agent.losses import USFALearning, usfa_loss_kwargs
# from acme.agents.jax.r2d2.networks import make_atari_networks
from agents.td_agent.utils import make_networks
from agents.td_agent.utils import make_behavior_policy
from agents.td_agent.networks import TDNetworkFns, R2D2Network, USFANetwork


