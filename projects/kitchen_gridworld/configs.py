"""Config."""
import dataclasses

from acme.adders import reverb as adders_reverb
from projects.common.configs import (
  R2D1Config,
  USFAConfig,
  QAuxConfig,
  RewardConfig,
  FarmConfig,
  ModularUSFAConfig,
  FarmModelConfig,
  NoiseConfig
)


@dataclasses.dataclass
class LangConfig:
  max_vocab_size: int = 30
  word_dim: int = 128  # dimension of both word and task (sentence) embeddings
  word_initializer: str = 'RandomNormal'
  word_compress: str = 'last'
