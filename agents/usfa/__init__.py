"""Implementation of an USFA agent."""

# from acme.agents.jax.r2d2.agents import DistributedR2D2
# from acme.agents.jax.r2d2.agents import DistributedR2D2FromConfig
from agents.usfa.agents import USFA
from agents.usfa.builder import USFABuilder
from agents.usfa.config import Config
# from acme.agents.jax.r2d2.learning import R2D2Learner
# from acme.agents.jax.r2d2.networks import make_atari_networks
from agents.usfa.utils import make_behavior_policy
from agents.usfa.utils import make_usfa_networks
from agents.usfa.networks import USFANetwork


