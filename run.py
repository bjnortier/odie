import sys
import os.path as osp
sys.path.insert(0, osp.realpath(osp.join('..', 'baselines')))
sys.path.insert(0, osp.realpath(osp.join('..', 'gym')))
# sys.path.insert(0, osp.realpath(osp.join('..', 'mujoco-py')))
import gym
import gym.spaces

gym.envs.register(
    id='Odie-v2',
    entry_point='envs.mujoco:make_odie_v2',
    max_episode_steps=1024,
    reward_threshold=4800.0,
)

from baselines.run import main
from minotaur import create_minotaur_experiment

if __name__ == '__main__':
    create_minotaur_experiment()
    main()
