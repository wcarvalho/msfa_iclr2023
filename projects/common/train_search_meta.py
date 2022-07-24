"""
Param search.
"""

import os
from absl import app
import time
from pprint import pprint
from absl import flags
import subprocess

flags.DEFINE_spaceseplist('searches', '', 'which search to use.')
flags.DEFINE_string('python_file', '', 'which python script to use.')

from projects.common import train_distributed
from projects.common import train_search

def main(_):
  """This will loop through the spaces list in FLAGS.spaces and run each item on a different GPU
  
  Args:
      _ (TYPE): Description
  """
  FLAGS = flags.FLAGS
  def build_command(search=None, idx=None):
    command = f"""python {FLAGS.python_file}
      --wandb={FLAGS.wandb}
      --wandb_project={FLAGS.wandb_project}
      --folder={FLAGS.folder}
      --group={FLAGS.group}
      --notes={FLAGS.notes}
      --date={FLAGS.date}
      --agent={FLAGS.agent}
      --env={FLAGS.env}
      --spaces={FLAGS.spaces}
      --search={search}
      --terminal={FLAGS.terminal}
      --skip={FLAGS.skip}
      --idx={idx}
      --ray={FLAGS.ray}
      --debug_search={FLAGS.debug_search}
      """

    if search is not None:
      command += f" --search={search}"

    if idx is not None:
      command += f" --idx={idx}"

    return command

  def run(command, cuda):
    pprint(command)
    command = command.replace("\n", '')

    cuda_env = os.environ.copy()
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    return subprocess.Popen(command, env=cuda_env, shell=True)

  gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

  if len(FLAGS.searches) > 1:
    for idx, search in enumerate(FLAGS.searches):
      cuda = gpus[idx%len(gpus)]
      command = build_command(search=search)
      p = run(command, cuda)
      if not FLAGS.debug_search:
        time.sleep(10.0)
  else:
    for idx in range(len(gpus)):
      cuda = gpus[idx%len(gpus)]
      command = build_command(search=FLAGS.searches[0], idx=idx)
      p = run(command, cuda)
      if not FLAGS.debug_search:
        time.sleep(10.0)

if __name__ == '__main__':
  app.run(main)
