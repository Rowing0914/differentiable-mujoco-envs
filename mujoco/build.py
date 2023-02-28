from mujoco.utils.wrappers.mj_block import MjBlockWrapper
from mujoco.utils.wrappers.etc import SnapshotWrapper, IndexWrapper


def build_env(env_name, batch_size):
    if env_name == "HopperEnv":
        from mujoco.envs.hopper import HopperEnv as env_cls
    elif env_name == "HalfCheetahEnv":
        from mujoco.envs.half_cheetah import HalfCheetahEnv as env_cls
    elif env_name == "SwimmerEnv":
        from mujoco.envs.swimmer import SwimmerEnv as env_cls
    elif env_name == "InvertedPendulumEnv":
        from mujoco.envs.inverted_pendulum import InvertedPendulumEnv as env_cls
    elif env_name == "InvertedDoublePendulumEnv":
        from mujoco.envs.inverted_double_pendulum import InvertedDoublePendulumEnv as env_cls
    elif env_name == "Walker2dEnv":
        from mujoco.envs.walker2d import Walker2dEnv as env_cls
    else:
        raise ValueError

    env = env_cls()

    # Record video
    # env = ViewerWrapper(env)

    # Keep track of step, episode, and batch indices
    env = IndexWrapper(env, batch_size)

    # Grab and set snapshots of data
    env = SnapshotWrapper(env)

    # This should probably be last so we get all wrappers
    env = MjBlockWrapper(env)
    return env
