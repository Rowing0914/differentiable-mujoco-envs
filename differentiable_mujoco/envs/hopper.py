import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    COPIED FROM GYM. W/ SLIGHT MODIFICATIONS:
        * READING FROM OWN .XML.
        * FULL STATE OBSERVATIONS, I.E. QPOS CONCAT'D WITH QVEL.
        * is_done METHOD SHOULD BE IMPLEMENTED
        * torch implementation of reward function
    """

    def __init__(self, frame_skip=20):
        # self.cfg = cfg
        self.frame_skip = frame_skip
        mujoco_assets_dir = os.path.abspath("./gen_rl/envs/differentiable_mujoco/assets/")
        self.initialised = False
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "hopper.xml"), self.frame_skip)
        utils.EzPickle.__init__(self)
        self.initialised = True

    def sigmoid(self, x, mi, mx):
        return mi + (mx - mi) * (lambda t: (1 + 200 ** (-t + 0.5)) ** (-1))((x - mi) / (mx - mi))

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        ang_abs = abs(ang) % (2 * math.pi)
        if ang_abs > math.pi:
            ang_abs = 2 * math.pi - ang_abs
        coeff1 = self.sigmoid(height / 1.25, 0, 1)
        coeff2 = self.sigmoid((math.pi - ang_abs) / math.pi, 0, 1)
        reward += coeff1 * alive_bonus + coeff2 * alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2)) and self.initialised
        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        """DIFFERENT FROM ORIGINAL GYM"""
        return np.concatenate([
            self.sim.data.qpos.flat,  # this part different from gym. expose the whole thing.
            self.sim.data.qvel.flat,  # this part different from gym. clip nothing.
        ])

    def reset_model(self):
        self.sim.reset()
        # if self.cfg.MODEL.POLICY.NETWORK:
        #     qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        #     qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        #     self.set_state(qpos, qvel)
        # else:
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    @staticmethod
    def is_done(state):
        height, ang = state[1:3]
        done = not (np.isfinite(state).all() and (np.abs(state[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        return done
