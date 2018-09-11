import sys
sys.path.insert(0, "/Users/bjnortier/development/RL/baselines")
sys.path.insert(0, "/Users/bjnortier/development/RL/gym")
sys.path.insert(0, "/Users/bjnortier/development/RL/mujoco-py")
import gym
import gym.spaces

gym.envs.register(
    id='Odie-v2',
    entry_point='envs.mujoco:make_odie_v2',
    max_episode_steps=1024,
    reward_threshold=4800.0,
)

from baselines.run import main

if __name__ == '__main__':
    main()
