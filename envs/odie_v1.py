from os import path
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class OdieV1Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = path.join(path.dirname(__file__), 'assets', 'odie_v1.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

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

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
