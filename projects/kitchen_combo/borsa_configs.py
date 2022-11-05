from projects.msf.configs import *
from projects.msf import configs


@dataclasses.dataclass
class USFAConfig(configs.USFAConfig):
  npolicies: int = 1

@dataclasses.dataclass
class ModularUSFAConfig(USFAConfig, configs.ModularUSFAConfig):
  npolicies: int = 1
  memory_size: int = None
  module_size: int = 150
  nmodules: int = 4


@dataclasses.dataclass
class FarmConfig(configs.FarmConfig):
  module_size: int = 150
  nmodules: int = 4
  memory_size: int = None