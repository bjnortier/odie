import math
from os import path
import numpy as np
import mujoco_py
from mujoco_py.generated import const
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.wrappers.monitoring import video_recorder
from minotaur import MinotaurMonitor

class PiperV2Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = path.join(path.dirname(__file__), 'assets', 'piper_v2.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        try:
            self.do_simulation(action, self.frame_skip)
            xposafter = self.sim.data.qpos[0]
            torso_pitch_angle_after = self.sim.data.qpos[2]
            reward_ctrl = - 0.1 * np.square(action).sum()
            reward_run = (xposafter - xposbefore)/self.dt
            reward_torso_pitch = -np.square(torso_pitch_angle_after)
            reward = reward_ctrl + reward_run + reward_torso_pitch
            ob = self._get_obs()
            done = False
            return ob, reward, done, dict()
        except mujoco_py.builder.MujocoException as e:
            ob = self._get_obs()
            reward = -1000
            done = True
            return ob, reward, done, dict()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        # First 3 values are the torso joints (translation x 2 rotation x 1)
        # Have to be careful that the legs do not start inside the ground groundplane
        assert(self.model.nq == 11)
        qpos = np.concatenate([
            np.zeros(4),
            self.np_random.uniform(low=-0.4, high=0.4, size=self.model.nq - 4)
        ])

        # qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 2

def make_piper_v2():
    env = PiperV2Env()
    return MinotaurMonitor('Piper-v2', env)
