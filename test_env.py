import sys
import os.path as osp
import numpy as np
sys.path.insert(0, osp.realpath(osp.join('..', 'baselines')))
sys.path.insert(0, osp.realpath(osp.join('..', 'gym')))
sys.path.insert(0, osp.realpath(osp.join('..', 'mujoco-py')))
import gym
import gym.spaces

gym.envs.register(
    id='Piper-v2',
    entry_point='envs.mujoco:make_piper_v2',
    max_episode_steps=1024,
    reward_threshold=4096.0,
)

env = gym.make('Piper-v2')
env.reset()
while True:
    env.env.env.do_simulation(np.zeros(8), 1)
    env.render()
