"""Loggeres."""

import time
from typing import Optional
from absl import logging

from acme.utils import loggers
from acme.utils.loggers import base
from acme.utils.loggers import asynchronous as async_logger
from acme.jax import utils as jax_utils
import jax
import numpy as np
from utils.tf_summary import TFSummaryLogger
from utils import data as data_utils

import datetime
from pathlib import Path

try:
  from rich import print as rich_print
except ImportError:
  rich_print = None

try:
  import wandb
  WANDB_AVAILABLE=True
except ImportError:
  WANDB_AVAILABLE=False

def gen_log_dir(base_dir="results/", date=True, hourminute=True, seed=None, return_kwpath=False, path_skip=[], **kwargs):

  strkey = '%Y.%m.%d'
  if hourminute:
    strkey += '-%H.%M'
  job_name = datetime.datetime.now().strftime(strkey)
  kwpath = ','.join([f'{key[:4]}={value}' for key, value in kwargs.items() if not key in path_skip])

  if date:
    path = Path(base_dir).joinpath(job_name).joinpath(kwpath)
  else:
    path = Path(base_dir).joinpath(kwpath)

  if seed is not None:
    path = path.joinpath(f'seed={seed}')

  if return_kwpath:
    return str(path), kwpath
  else:
    return str(path)

def copy_numpy(values):
  return jax.tree_map(np.array, values)

class FlattenFilter(base.Logger):
  """"""

  def __init__(self, to: base.Logger):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
    """
    self._to = to

  def write(self, values: base.LoggingData):
    values = data_utils.flatten_dict(values, sep='/')
    self._to.write(values)

  def close(self):
    self._to.close()

def make_logger(
  log_dir: str,
  label: str,
  save_data: bool = True,
  asynchronous: bool = False,
  tensorboard=True,
  wandb=False,
  time_delta: float=10.0,
  steps_key: str=None) -> loggers.Logger:
  """Creates ACME loggers as we wish.
  Features:
    - CSV Logger
    - TF Summary
  """
  # See acme.utils.loggers.default:make_default_logger
  wandb = wandb and WANDB_AVAILABLE
  _loggers = [
      loggers.TerminalLogger(label=label, print_fn=rich_print or print),
  ]

  if save_data:
    _loggers.append(loggers.CSVLogger(log_dir, label=label, add_uid=False))



  if tensorboard:
    _loggers.append(
      TFSummaryLogger(log_dir, label=label, steps_key=steps_key))

  if wandb:
    _loggers.append(WandbLogger(label=label, steps_key=steps_key))

  # Dispatch to all writers and filter Nones.
  logger = loggers.Dispatcher(_loggers, copy_numpy)  # type: ignore
  logger = loggers.NoneFilter(logger)
  logger = FlattenFilter(logger)

  if asynchronous:
    logger = async_logger.AsyncLogger(logger)

  # filter by time: Print logs almost every 10 seconds.
  if time_delta:
    logger = loggers.TimeFilter(logger, time_delta=time_delta)


  return logger


def _format_key(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  new = key.title().replace("_", "").replace("/", "-")
  return new

def _format_loss(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  new = key.title().replace("_", "")
  return new



class WandbLogger(base.Logger):
  """Logs to a tf.summary created in a given logdir.

  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
      self,
      label: str = 'Logs',
      labels_skip=('Loss'),
      steps_key: Optional[str] = None
  ):
    """Initializes the logger.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
      steps_key: key to use for steps. Must be in the values passed to write.
    """
    self._time = time.time()
    self.label = label
    self.labels_skip =labels_skip
    self._iter = 0
    # self.summary = tf.summary.create_file_writer(logdir)
    self._steps_key = steps_key

  def write(self, values: base.LoggingData):
    if self._steps_key is not None and self._steps_key not in values:
      logging.warn('steps key "%s" not found. Skip logging.', self._steps_key)
      logging.warn('Available keys:', str(values.keys()))
      return

    step = values[self._steps_key] if self._steps_key is not None else self._iter


    to_log={}
    for key in values.keys() - [self._steps_key]:

      if self.label in self.labels_skip: # e.g. [Loss]
        key_pieces = key.split("/")
        if len(key_pieces) == 1: # e.g. [step]
          name = f'{self.label}/{_format_key(key)}'
        else: 
          if 'grad' in key.lower():
          # e.g. [MeanGrad/FarmSharedOutput/~/FeatureAttention/Conv2D1] --> [Loss/MeanGrad-FarmSharedOutput-~-FeatureAttention-Conv2D1]
            name = f'grads/{_format_key(key)}'
          else: # e.g. [r2d1/xyz] --> [Loss_r2d1/xyz]
            name = f'{self.label}_{_format_loss(key)}'
      else: # e.g. [actor_SmallL2NoDist]
        name = f'{self.label}/{_format_key(key)}'

      to_log[name] = values[key]

    to_log[f'{self.label}/step']  = step


    wandb.log(to_log)
    self._iter += 1

  def close(self):
    wandb.finish()

