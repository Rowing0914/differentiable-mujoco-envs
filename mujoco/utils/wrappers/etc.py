import gym
import numpy as np
from copy import deepcopy
import mujoco_py


class SnapshotWrapper(gym.Wrapper):
    """Handles all stateful stuff, like getting and setting snapshots of states, and resetting"""

    def get_snapshot(self):
        class DataSnapshot:
            # Note: You should not modify these parameters after creation

            def __init__(self, d_source, step_idx):
                self.time = deepcopy(d_source.time)
                self.qpos = deepcopy(d_source.qpos)
                self.qvel = deepcopy(d_source.qvel)
                self.qacc_warmstart = deepcopy(d_source.qacc_warmstart)
                self.ctrl = deepcopy(d_source.ctrl)
                self.act = deepcopy(d_source.act)
                self.qfrc_applied = deepcopy(d_source.qfrc_applied)
                self.xfrc_applied = deepcopy(d_source.xfrc_applied)

                self.step_idx = deepcopy(step_idx)

                # These probably aren't necessary, but they should fix the body in the same position with
                # respect to worldbody frame?
                self.body_xpos = deepcopy(d_source.body_xpos)
                self.body_xquat = deepcopy(d_source.body_xquat)

        return DataSnapshot(self.env.sim.data, self.get_step_idx())

    def set_snapshot(self, snapshot_data):
        self.env.sim.data.time = deepcopy(snapshot_data.time)
        self.env.sim.data.qpos[:] = deepcopy(snapshot_data.qpos)
        self.env.sim.data.qvel[:] = deepcopy(snapshot_data.qvel)
        self.env.sim.data.qacc_warmstart[:] = deepcopy(snapshot_data.qacc_warmstart)
        self.env.sim.data.ctrl[:] = deepcopy(snapshot_data.ctrl)
        if snapshot_data.act is not None:
            self.env.sim.data.act[:] = deepcopy(snapshot_data.act)
        self.env.sim.data.qfrc_applied[:] = deepcopy(snapshot_data.qfrc_applied)
        self.env.sim.data.xfrc_applied[:] = deepcopy(snapshot_data.xfrc_applied)

        self.set_step_idx(snapshot_data.step_idx)

        self.env.sim.data.body_xpos[:] = deepcopy(snapshot_data.body_xpos)
        self.env.sim.data.body_xquat[:] = deepcopy(snapshot_data.body_xquat)


class Index(object):
    def __init__(self, value=0):
        self.value = value

    def __index__(self):
        return self.value

    def __iadd__(self, other):
        self.value += other
        return self

    def __add__(self, other):
        if isinstance(other, Index):
            return Index(self.value + other.value)
        elif isinstance(other, int):
            return Index(self.value + other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Index):
            return Index(self.value - other.value)
        elif isinstance(other, int):
            return Index(self.value - other)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        return self.value == other

    def __repr__(self):
        return "Index({})".format(self.value)

    def __str__(self):
        return "{}".format(self.value)

    def set(self, other):
        if isinstance(other, int):
            self.value = other
        elif isinstance(other, Index):
            self.value = other.value
        else:
            raise NotImplementedError


class IndexWrapper(gym.Wrapper):
    """Counts steps and episodes"""

    def __init__(self, env, batch_size):
        super(IndexWrapper, self).__init__(env)
        self._step_idx = Index(0)
        self._episode_idx = Index(0)
        self._batch_idx = Index(0)
        self._batch_size = batch_size

        # Add references to the unwrapped env because we're going to need at least step_idx
        # NOTE! This means we can never overwrite self.step_idx, self.episode_idx, or self.batch_idx or we lose
        # the reference
        env.unwrapped._step_idx = self._step_idx
        env.unwrapped._episode_idx = self._episode_idx
        env.unwrapped._batch_idx = self._batch_idx

    def step(self, action):
        self._step_idx += 1
        return self.env.step(action)

    def reset(self, update_episode_idx=True):
        self._step_idx.set(0)

        # We don't want to update episode_idx during testing
        if update_episode_idx:
            if self._episode_idx == self._batch_size:
                self._batch_idx += 1
                self._episode_idx.set(1)
            else:
                self._episode_idx += 1

        return self.env.reset()

    def get_step_idx(self):
        return self._step_idx

    def get_episode_idx(self):
        return self._episode_idx

    def get_batch_idx(self):
        return self._batch_idx

    def set_step_idx(self, idx):
        self._step_idx.set(idx)

    def set_episode_idx(self, idx):
        self._episode_idx.set(idx)

    def set_batch_idx(self, idx):
        self._batch_idx.set(idx)


class ViewerWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ViewerWrapper, self).__init__(env)

        # Keep params in this class to reduce clutter
        class Recorder:
            width = 1600
            height = 1200
            imgs = []
            record = False
            filepath = None

        self.recorder = Recorder

        # Check if we want to record roll-outs
        if self.cfg.LOG.TESTING.RECORD_VIDEO:
            self.recorder.record = True

        # Create a viewer if we're not recording
        else:
            # Initialise a MjViewer
            self._viewer = mujoco_py.MjViewer(self.sim)
            self._viewer._run_speed = 1 / self.cfg.MODEL.FRAME_SKIP
            self.unwrapped._viewers["human"] = self._viewer

    def capture_frame(self):
        if self.recorder.record:
            self.recorder.imgs.append(np.flip(self.sim.render(self.recorder.width, self.recorder.height), axis=0))

    def start_recording(self, filepath):
        if self.recorder.record:
            self.recorder.filepath = filepath
            self.recorder.imgs.clear()

    def stop_recording(self):
        if self.recorder.record:
            writer = skvideo.io.FFmpegWriter(
                self.recorder.filepath, inputdict={"-s": "{}x{}".format(self.recorder.width, self.recorder.height),
                                                   "-r": str(
                                                       1 / (self.model.opt.timestep * self.cfg.MODEL.FRAME_SKIP))})
            for img in self.recorder.imgs:
                writer.writeFrame(img)
            writer.close()
            self.recorder.imgs.clear()
