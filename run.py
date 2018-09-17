import sys
import os.path as osp
sys.path.insert(0, osp.realpath(osp.join('..', 'baselines')))
sys.path.insert(0, osp.realpath(osp.join('..', 'gym')))
# sys.path.insert(0, osp.realpath(osp.join('..', 'mujoco-py')))
import gym
import gym.spaces
from baselines.common.cmd_util import common_arg_parser

gym.envs.register(
    id='Odie-v2',
    entry_point='envs.mujoco:make_odie_v2',
    max_episode_steps=1024,
    reward_threshold=4800.0,
)

gym.envs.register(
    id='Piper-v1',
    entry_point='envs.mujoco:make_piper_v1',
    max_episode_steps=1024,
    reward_threshold=4096.0,
)

gym.envs.register(
    id='Piper-v2',
    entry_point='envs.mujoco:make_piper_v2',
    max_episode_steps=1024,
    reward_threshold=4096.0,
)

from baselines.run import main
from minotaur import create_minotaur_experiment

if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    create_minotaur_experiment(args.env)
    main()
