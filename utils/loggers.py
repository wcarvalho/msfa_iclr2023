"""Loggeres."""

from acme.utils import loggers
from acme.utils.loggers import tf_summary
from acme.utils.loggers import asynchronous as async_logger

import datetime
from pathlib import Path

try:
  from rich import print as rich_print
except ImportError:
  rich_print = None

def gen_log_dir(base_dir="results/", **kwargs):
    job_name = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M')
    kwpath = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    path = Path(base_dir).joinpath(job_name).joinpath(kwpath)
    return str(path)


def make_logger(
  log_dir: str,
  label: str,
  save_data: bool = True,
  asynchronous: bool = False,
  steps_key: str=None) -> loggers.Logger:
  """Creates ACME loggers as we wish.
  Features:
    - CSV Logger
    - TF Summary
  """
  # See acme.utils.loggers.default:make_default_logger

  _loggers = [
      loggers.TerminalLogger(label=label, print_fn=rich_print or print),
  ]
  if save_data:
    _loggers.append(loggers.CSVLogger(log_dir, label=label, add_uid=False))
  
  _loggers.append(
    tf_summary.TFSummaryLogger(log_dir, label=label, steps_key=steps_key))

  # Dispatch to all writers and filter Nones.
  logger = loggers.Dispatcher(_loggers, loggers.to_numpy)  # type: ignore
  logger = loggers.NoneFilter(logger)

  if asynchronous:
    logger = async_logger.AsyncLogger(logger)

  # filter by time: Print logs almost every 10 seconds.
  logger = loggers.TimeFilter(logger, time_delta=10.0)
  return logger

