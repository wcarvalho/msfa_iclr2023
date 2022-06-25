"""
THIS FILE DOESN'T WORK, FEEL FREE TO IGNORE IT




Run Successor Feature based agents and baselines on
  BabyAI derivative environments.

Command I run to train:
  PYTHONPATH=$PYTHONPATH:$HOME/successor_features/rljax/ \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/ \
    CUDA_VISIBLE_DEVICES=3 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    WANDB_START_METHOD="thread" \
    python projects/colocation/run_many_seeds.py \
    --agent usfa_conv --room_reward .25 --train_task_as_z 1 --num_seeds 3 --wandb_name 6-5

Command for tensorboard
ssh -L 16006:127.0.0.1:6006 nameer@deeplearn9.eecs.umich.edu
source ~/.bashrc; conda activate acmejax; cd ~/successor_features/rljax/results/colocation/distributed/experiments
tensorboard --logdir .
"""
#most recent 2 are r2d1 and then r2d1_noise, both with no walls, 6 objects

# Do not preallocate GPU memory for JAX.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print(os.environ['LD_LIBRARY_PATH'])


import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from absl import app
from absl import flags
import time
from projects.colocation.train_distributed import build_program



# -----------------------
# flags
# -----------------------

flags.DEFINE_integer('num_seeds', 1, 'How many seeds to run') #the only new one

FLAGS = flags.FLAGS

def main(_):
    for seed in range(FLAGS.num_seeds):
      config_kwargs = dict(seed=seed)

      if FLAGS.max_number_of_steps is not None:
          config_kwargs['max_number_of_steps'] = FLAGS.max_number_of_steps

      wandb_init_kwargs = dict(
          project=FLAGS.wandb_project,
          entity=FLAGS.wandb_entity,
          group=FLAGS.agent,  # organize individual runs into larger experiment
          notes=FLAGS.wandb_notes
      )

      program = build_program(
          agent=FLAGS.agent,
          num_actors=FLAGS.num_actors,
          config_kwargs=config_kwargs,
          wandb_init_kwargs=wandb_init_kwargs if FLAGS.wandb else None,
          simple=FLAGS.simple,
          nowalls=FLAGS.nowalls,
          one_room=FLAGS.one_room,
          deterministic_rooms=FLAGS.deterministic_rooms,
          room_reward=FLAGS.room_reward,
          wandb_name=FLAGS.wandb_name,
          train_task_as_z=FLAGS.train_task_as_z,
          randomize_name=False
      )

      # Launch experiment.
      controller = lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING,
                             terminal='current_terminal',
                             local_resources={
                                 'actor':
                                     PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
                                 'evaluator':
                                     PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
                             )
      time.sleep(60)
      controller.wait()
      time.sleep(60)

if __name__ == '__main__':
  app.run(main)
