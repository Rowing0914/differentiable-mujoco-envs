from differentiable_mujoco.utils.wrappers.mj_block import MjBlockWrapper
from differentiable_mujoco.utils.wrappers.etc import SnapshotWrapper, IndexWrapper


def build_env(env_name, batch_size, max_episode_steps):
    if env_name == "Hopper":
        from differentiable_mujoco.envs.hopper import HopperEnv as env_cls
    elif env_name == "HalfCheetah":
        from differentiable_mujoco.envs.half_cheetah import HalfCheetahEnv as env_cls
    elif env_name == "SwimmerEnv":
        from differentiable_mujoco.envs.swimmer import SwimmerEnv as env_cls
    elif env_name == "InvertedPendulum":
        from differentiable_mujoco.envs.inverted_pendulum import InvertedPendulumEnv as env_cls
    elif env_name == "InvertedDoublePendulum":
        from differentiable_mujoco.envs.inverted_double_pendulum import InvertedDoublePendulumEnv as env_cls
    elif env_name == "Walker2d":
        from differentiable_mujoco.envs.walker2d import Walker2dEnv as env_cls
    else:
        raise ValueError

    env = env_cls(max_episode_steps=max_episode_steps)
    # from gym.wrappers.time_limit import TimeLimit
    # env = TimeLimit(env=env, max_episode_steps=max_episode_steps)

    # Record video
    # env = ViewerWrapper(env)

    # Keep track of step, episode, and batch indices
    env = IndexWrapper(env, batch_size)

    # Grab and set snapshots of data
    env = SnapshotWrapper(env)

    # This should probably be last so we get all wrappers
    env = MjBlockWrapper(env)
    return env
