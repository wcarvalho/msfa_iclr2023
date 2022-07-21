class BaseLoss:
  """docstring for BaseLoss"""
  def __init__(self, random=False, elementwise=False, **kwargs):
    self.elementwise = elementwise
    self.random = random
