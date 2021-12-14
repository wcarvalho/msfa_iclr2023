import yaml
import ipdb
import cv2
import numpy as np
from envs.babyaikitchen import babyai_utils
from envs.babyaikitchen.multilevel import KitchenMultiLevel
from envs.babyaikitchen.wrappers import RGBImgPartialObsWrapper
import gym_minigrid.window


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tasks', help='tasks file', default='tasks/babyai_kitchen/unseen_arg/length=3_cook.yaml')
    parser.add_argument('--sets', help='sets file',
        default="tasks/babyai_kitchen/default_sets.yaml")
    parser.add_argument('--num-missions', help='# of unique missions',
        default=10)
    parser.add_argument('--room-size', type=int, default=8)
    parser.add_argument('--agent-view-size', type=int, default=7)
    parser.add_argument('--render-mode', type=str, default='human')
    parser.add_argument('--actions', type=str, 
        default=['left', 'right', 'forward', 'pickup_contents',
                 'pickup_container', 'place', 'toggle', 'slice'],
        nargs="+")
    parser.add_argument('--random-object-state', type=int, default=0)
    parser.add_argument('--num-rows', type=int, default=1)
    parser.add_argument('--tile-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--show-both', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--check', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=2)
    args = parser.parse_args()

    # -----------------------
    # load file w/ sets of objects
    # -----------------------
    with open(args.sets, 'r') as f:
      sets = yaml.load(f, Loader=yaml.SafeLoader)

    # -----------------------
    # load file w/ tasks 
    # -----------------------
    with open(args.tasks, 'r') as f:
      tasks = yaml.load(f, Loader=yaml.SafeLoader)

    level_kwargs=dict(
        room_size=args.room_size,
        agent_view_size=args.agent_view_size,
        random_object_state=args.random_object_state,
        actions=args.actions,
        verbosity=args.verbosity,
        tile_size=args.tile_size,
        use_time_limit=False,
        seed=args.seed,
        )
    if args.num_rows:
        level_kwargs['num_rows'] = args.num_rows
        level_kwargs['num_cols'] = args.num_rows

    level_kwargs = babyai_utils.constuct_kitchenmultilevel_kwargs(
        task_dicts=tasks['train'],
        level_kwargs=level_kwargs,
        sets=sets)


    env = KitchenMultiLevel(level_kwargs)
    # mimic settings during training
    env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)
    render_kwargs = {'tile_size' : env.tile_size}

    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)

    def combine(full, partial):
        if args.show_both:
            full_small = cv2.resize(full, dsize=partial.shape[:2], 
                interpolation=cv2.INTER_CUBIC)
            return np.concatenate((full_small, partial), axis=1)
        else:
            return full


    def forward():
        obs, _, _, _ = env.step(2); 
        full = env.render('rgb_array')
        window.show_img(combine(full, obs['image']))
    def left():
        obs, _, _, _ = env.step(0); 
        full = env.render('rgb_array')
        window.show_img(combine(full, obs['image']))
    def right():
        obs, _, _, _ = env.step(1); 
        full = env.render('rgb_array')
        window.show_img(combine(full, obs['image']))

    for mission_indx in range(int(args.num_missions)):
        env.seed(mission_indx)
        obs = env.reset()
        print("="*50)
        print("Reset")
        print("="*50)
        print("Level:", obs['level'])
        print("Task:", obs['mission'])
        print("Image Shape:", obs['image'].shape)


        full = env.render('rgb_array', **render_kwargs)
        window.set_caption(obs['mission'])
        window.show_img(combine(full, obs['image']))

        for step in range(args.steps):
            obs, reward, done, info = env.step(env.action_space.sample())


            full = env.render('rgb_array', **render_kwargs)
            window.set_caption(obs['mission'])
            window.show_img(combine(full, obs['image']))

            if done:
                print(f"Complete! Reward: {reward}")
                print(f"info: {str(info)}")
                print(f"Episode length: {step+1}")
                break
        if int(args.check):
            print("print 'c' to contnue")
            ipdb.set_trace()


if __name__ == "__main__":
    main()
