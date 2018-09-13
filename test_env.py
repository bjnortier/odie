from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import mujoco_py

from os.path import dirname, join
model = mujoco_py.load_model_from_path(join(dirname(__file__), 'envs', 'mujoco', 'assets', 'odie_v2.xml'))
# model = mujoco_py.load_model_from_path(join(dirname(__file__), '..', '..', 'bots', 'max', 'src', 'envs', 'assets', 'metalhead_v4.xml'))

sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
# sim.data.qvel[1] = 0.1
while True:
    sim.data.ctrl[0] = math.cos(t / 100.) * 1
    sim.data.ctrl[1] = math.sin(t / 100.) * 1
    sim.data.ctrl[2] = math.cos(t / 100.) * 1
    sim.data.ctrl[3] = math.sin(t / 100.) * 1
    # sim.data.ctrl[4] = math.cos(t / 100.) * 1
    # sim.data.ctrl[5] = math.sin(t / 100.) * 1
    # sim.data.ctrl[6] = math.cos(t / 100.) * 1
    # sim.data.ctrl[7] = math.sin(t / 100.) * 1
    print(sim.data.ctrl)
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
