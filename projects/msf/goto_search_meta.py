"""
Param search.
"""

import os
from absl import app
from pprint import pprint
from absl import flags
import subprocess
from projects.msf.goto_distributed import build_program
from projects.msf.goto_search_lp import main

flags.DEFINE_spaceseplist('searches', 'baselines', 'which search to use.')

def main(_):
  """This will loop through the spaces list in FLAGS.spaces and run each item on a different GPU
  
  Args:
      _ (TYPE): Description
  """
  FLAGS = flags.FLAGS
  def build_command(search=None, idx=None):
    command = f"""python projects/msf/goto_search_lp.py
      --folder={FLAGS.folder}
      --wandb={FLAGS.wandb}
      --date={FLAGS.date}
      --spaces={FLAGS.spaces}
      --wandb_project={FLAGS.wandb_project}
      --group={FLAGS.group}
      --notes={FLAGS.notes}
      --skip={FLAGS.skip}
      --ray={FLAGS.ray}
      --debug_search={FLAGS.debug_search}
      --agent={FLAGS.agent}"""
    command = command.replace("\n", '')

    if search is not None:
      command += f" --search={search}"

    if idx is not None:
      command += f" --idx={idx}"

    return command

  def run(command, cuda):
    pprint(command)
    cuda_env = os.environ.copy()
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    return subprocess.Popen(command, env=cuda_env, shell=True)

  print(FLAGS.searches)
  gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

  if len(FLAGS.searches) > 1:
    for idx, search in enumerate(FLAGS.searches):
      cuda = gpus[idx%len(gpus)]
      command = build_command(search=search)
      p = run(command, cuda)
  else:
    for idx in range(len(gpus)):
      cuda = gpus[idx%len(gpus)]
      command = build_command(search=FLAGS.searches[0], idx=idx)
      p = run(command, cuda)



if __name__ == '__main__':
  app.run(main)
