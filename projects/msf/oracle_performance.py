import gym_minigrid.window
from envs.babyai_kitchen.bot import GotoAvoidBot

from envs.babyai_kitchen.wrappers import RGBImgPartialObsWrapper, RGBImgFullyObsWrapper
from envs.babyai_kitchen.goto_avoid import GotoAvoidEnv
import matplotlib.pyplot as plt 
import cv2
import tqdm
import numpy as np

def main():

  eval_tasks = {
      # "1,0,0,0":{
      #     "pan" : 1,
      #     "plates" : 0,
      #     "tomato" : 0,
      #     "knife" : 0,
      #     },
      # "0,1,0,0":{
      #     "pan" : 0,
      #     "plates" : 1,
      #     "tomato" : 0,
      #     "knife" : 0,
      #     },
      # "0,0,1,0":{
      #     "pan" : 0,
      #     "plates" : 0,
      #     "tomato" : 1,
      #     "knife" : 0,
      #     },
      # "0,0,0,1":{
      #     "pan" : 0,
      #     "plates" : 0,
      #     "tomato" : 0,
      #     "knife" : 1,
      #     },
      # '1,1,0,0':{
      #     "pan" : 1,
      #     "plates" :1,
      #     "tomato" : 0,
      #     "knife" : 0,
      #     },
      '1,1,5,5':{
          "pan" : 1,
          "plates" :1,
          "tomato" : .5,
          "knife" : .5,
          },
      '-.5,1,-.5,-.5':{
          "pan" : -.5,
          "plates" :1,
          "tomato" : -.5,
          "knife" : -.5,
          },
      # '1,1,1,1':{
      #     "pan" : 1,
      #     "plates" : 1,
      #     "tomato" : 1,
      #     "knife" : 1,
      #     },
      # '-1,1,0,1':{
      #     "pan" : -1,
      #     "plates" : 1,
      #     "tomato" : 0,
      #     "knife" : 1,
      #     },
      # '-1,1,-1,1':{
      #     "pan" : -1,
      #     "plates" : 1,
      #     "tomato" : -1,
      #     "knife" : 1,
      #     },
      # '-1,1,-1,-1':{
      #     "pan" : -1,
      #     "plates" : 1,
      #     "tomato" : -1,
      #     "knife" : -1,
      #     }
  }

  tile_size=8
  num_evaluation=100
  tasks = len(eval_tasks)
  all_rewards = np.zeros((tasks, num_evaluation))

  window = gym_minigrid.window.Window('kitchen')
  window.show(block=False)

  def combine(full, partial):
      full_small = cv2.resize(full, dsize=partial.shape[:2], interpolation=cv2.INTER_CUBIC)
      return np.concatenate((full_small, partial), axis=1)

  def plot_fn(obs):
    full = env.render('rgb_array', tile_size=tile_size, highlight=True)
    window.set_caption(obs['mission'])
    window.show_img(combine(full, obs['image']))

  for t_i, (name, object2reward) in tqdm.tqdm(enumerate(eval_tasks.items()), desc="level"):

    env = GotoAvoidEnv(
        agent_view_size=9,
        object2reward=object2reward,
        respawn=True,
        pickup_required=True,
        tile_size=tile_size,
        room_size=10,
        nobjects=3
        )
    env = RGBImgPartialObsWrapper(env, tile_size=tile_size)
    for e_i in tqdm.tqdm(range(num_evaluation)):
      obs = env.reset()
      bot = GotoAvoidBot(env)
      obss, actions, rewards, dones = bot.generate_traj(plot_fn=plot_fn)
      if obss is None:
        idx = 0
        while obss is None:
          obs = env.reset()
          bot = GotoAvoidBot(env)
          obss, actions, rewards, dones = bot.generate_traj(plot_fn=plot_fn)
          idx += 1
          if idx > 1000:
            raise RuntimeError("too many impossibilities")
      all_rewards[t_i, e_i] = sum(rewards)

  import ipdb; ipdb.set_trace()


if __name__ == '__main__':
  main()