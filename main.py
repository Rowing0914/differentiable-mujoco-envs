import torch
import torch.nn as nn

from differentiable_mujoco.torch_block import mj_torch_block_factory
from differentiable_mujoco.utils.wrappers.mj_block import MjBlockWrapper
from differentiable_mujoco.utils.wrappers.etc import SnapshotWrapper, IndexWrapper


def build_env(env_name, batch_size):
    if env_name == "HopperEnv":
        from differentiable_mujoco.envs.hopper import HopperEnv as env_cls
    elif env_name == "HalfCheetahEnv":
        from differentiable_mujoco.envs.half_cheetah import HalfCheetahEnv as env_cls
    elif env_name == "SwimmerEnv":
        from differentiable_mujoco.envs.swimmer import SwimmerEnv as env_cls
    elif env_name == "InvertedPendulumEnv":
        from differentiable_mujoco.envs.inverted_pendulum import InvertedPendulumEnv as env_cls
    elif env_name == "InvertedDoublePendulumEnv":
        from differentiable_mujoco.envs.inverted_double_pendulum import InvertedDoublePendulumEnv as env_cls
    elif env_name == "Walker2dEnv":
        from differentiable_mujoco.envs.walker2d import Walker2dEnv as env_cls
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


class Basic(nn.Module):
    def __init__(self, env, max_ep_steps):
        super(Basic, self).__init__()
        self.dynamics_block = mj_torch_block_factory(env=env, mode="dynamics", max_ep_steps=max_ep_steps).apply
        self.reward_block = mj_torch_block_factory(env=env, mode="reward", max_ep_steps=max_ep_steps).apply

    def forward(self, state, action):
        next_state = self.dynamics_block(state, action)
        reward = self.reward_block(state, action)
        return next_state, reward


def main():
    device = "cpu"
    env_name = "InvertedPendulumEnv"
    batch_size = 8
    max_ep_steps = 50

    env = build_env(env_name=env_name, batch_size=batch_size)
    model = Basic(env, max_ep_steps=max_ep_steps)
    device = torch.device(device)
    model.to(device)
    model.train()

    state = torch.tensor(env.reset()).float()
    policy = torch.nn.Linear(env.observation_space.shape[-1], env.action_space.shape[-1])
    value_fn = torch.nn.Linear(env.observation_space.shape[-1] + env.action_space.shape[-1], 1)
    optim = torch.optim.Adam(policy.parameters())
    for step_idx in range(max_ep_steps):
        # action = env.action_space.sample()
        action = policy(state)
        state, reward = model(state, action)
        Q = value_fn(torch.cat([state.float(), action.float()], dim=-1))
        loss = reward + Q
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.item())
        asdf
        if env.is_done:
            break


if __name__ == "__main__":
    main()
