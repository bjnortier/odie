import sys
import math
import os.path as osp
import numpy as np
sys.path.insert(0, osp.realpath(osp.join('..', 'baselines')))
sys.path.insert(0, osp.realpath(osp.join('..', 'gym')))
sys.path.insert(0, osp.realpath(osp.join('..', 'mujoco-py')))
import gym
import gym.spaces

gym.envs.register(
    id='Piper-v3',
    entry_point='envs.mujoco:make_piper_v3',
    max_episode_steps=1024,
    reward_threshold=4096.0,
)

env = gym.make('Piper-v3')
env.reset()
t = 0
while True:
    action = np.ones(8) * math.cos(t / 100.)
    env.env.env.step(action)
    env.render()
    t += 1
