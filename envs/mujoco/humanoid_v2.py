import os
import numpy as np
from gym import utils
from gym.envs.mujoco import HumanoidEnv
from minotaur import MinotaurMonitor

def make_humanoid_v2():
    env = HumanoidEnv()
    return MinotaurMonitor('Humanoid-Minotaur-v2', env)
