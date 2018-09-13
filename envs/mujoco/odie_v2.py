from os import path
import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.wrappers.monitoring import video_recorder
from minotaur import MinotaurMonitor

class OdieV2Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = path.join(path.dirname(__file__), 'assets', 'odie_v2.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        try:
            self.do_simulation(action, self.frame_skip)
            xposafter = self.sim.data.qpos[0]
            ob = self._get_obs()
            reward_ctrl = - 0.1 * np.square(action).sum()
            reward_run = (xposafter - xposbefore)/self.dt
            reward = reward_ctrl + reward_run
            done = self.sim.data.qpos[2] < -40
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
        # First 6 values are the torso joints (translation x 3, rotation x 3)
        # Have to be carful that the legs do not start inside the ground groundplane
        assert(self.model.nq == 10)
        qpos = np.concatenate([
            np.zeros(6),
            self.np_random.uniform(low=-.1, high=.1, size=4)
        ])
        # qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def close(self):
        self.minotaur_close()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 2

def make_odie_v2():
    env = OdieV2Env()
    return MinotaurMonitor('Odie-v2', env)
