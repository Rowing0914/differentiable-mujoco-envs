import os
import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    COPIED FROM GYM. W/ SLIGHT MODIFICATIONS:
        * READING FROM OWN .XML.
        * FULL STATE OBSERVATIONS, I.E. QPOS CONCAT'D WITH QVEL.
        * is_done METHOD SHOULD BE IMPLEMENTED
    """
    def __init__(self, frame_skip=10):
        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        # self.cfg = cfg
        self.frame_skip = frame_skip
        self.initialised = False
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "half_cheetah.xml"), self.frame_skip)
        utils.EzPickle.__init__(self)
        self.initialised = True

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        """DIFFERENT FROM ORIGINAL GYM"""
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        self.sim.reset()
        # if self.cfg.MODEL.POLICY.NETWORK:
        #     self.set_state(
        #         self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
        #         self.init_qvel + self.np_random.randn(self.model.nv) * .1
        #     )
        # else:
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    @staticmethod
    def is_done(state):
        done = False
        return done
