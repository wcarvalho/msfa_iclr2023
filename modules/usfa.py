from functools import partial
from archs.base import Network
from archs.vision import VisionTorso

from typing import NamedTuple


class USFAPredictions(NamedTuple):
  q: jnp.ndarray
  sf: jnp.ndarray
  policy_zeds: jnp.ndarray


class UsfaFarmMixture(Network):
  """docstring for ClassName"""
  def __init__(self, **kwargs):
    vision_net = kwargs.pop('vision_net', VisionTorso())

    super(UsfaFarmMixture, self).__init__(
      vision_net=vision_net,
      **kwargs)
