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
flags.DEFINE_spaceseplist('python_file', '', 'which python script to use.')


def main(_):
  """This will loop through the spaces list in FLAGS.spaces and run each item on a different GPU
  
  Args:
      _ (TYPE): Description
  """
  FLAGS = flags.FLAGS

  print(FLAGS.searches)
  gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

  for idx, search in enumerate(FLAGS.searches):
    cuda = gpus[idx%len(gpus)]
    command = f"""python {python_file}
      --folder={FLAGS.folder}
      --wandb={FLAGS.wandb}
      --date={FLAGS.date}
      --spaces={FLAGS.spaces}
      --wandb_project={FLAGS.wandb_project}
      --group={FLAGS.group}
      --search={search}
      --notes={FLAGS.notes}
      --skip={FLAGS.skip}
      --idx={idx}
      --ray={FLAGS.ray}
      --agent={FLAGS.agent} """
    command = command.replace("\n", '')
    pprint(command)
    cuda_env = os.environ.copy()
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    p = subprocess.Popen(command, env=cuda_env, shell=True)


if __name__ == '__main__':
  app.run(main)
