"""
Param search.
"""

import os
from absl import app
from pprint import pprint
from absl import flags
import subprocess
from projects.kitchen_gridworld.train_distributed import build_program
from projects.msf.goto_search_lp import main

flags.DEFINE_spaceseplist('searches', 'baselines', 'which search to use.')


def main(_):
  FLAGS = flags.FLAGS


  print(FLAGS.searches)
  gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

  for idx, search in enumerate(FLAGS.searches):
    cuda = gpus[idx%len(gpus)]
    command = f"""python projects/kitchen_gridworld/train_hp_search_lp.py
      --folder={FLAGS.folder}
      --num_gpus={FLAGS.num_gpus}
      --num_cpus={FLAGS.num_cpus}
      --actors={FLAGS.actors}
      --wandb={FLAGS.wandb}
      --date={FLAGS.date}
      --wandb_project={FLAGS.wandb_project}
      --group={FLAGS.group}
      --search={search}
      --notes={FLAGS.notes}
      --skip={FLAGS.skip}
      --ray={FLAGS.ray}
      --agent={FLAGS.agent} """
    command = command.replace("\n", '')
    pprint(command)
    cuda_env = os.environ.copy()
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    subprocess.Popen(command, env=cuda_env, shell=True)


if __name__ == '__main__':
  app.run(main)
