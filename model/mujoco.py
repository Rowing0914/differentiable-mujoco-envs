import torch
from torch import autograd, nn
import numpy as np


def mj_torch_block_factory(env, mode, max_ep_steps):
    mj_forward = env.forward_factory(mode)
    mj_gradients = env.gradient_factory(mode)

    class MjBlock(autograd.Function):

        @staticmethod
        def forward(ctx, state, action):

            # Advance simulation or return reward
            if mode == "dynamics":
                # We need to get a deep copy of simulation data so we can return to this "snapshot"
                # (we can't deepcopy env.sim.data because some variables are unpicklable)
                # We'll calculate gradients in the backward phase of "reward"
                env.data.qpos[:] = state[:env.model.nq].detach().numpy().copy()
                env.data.qvel[:] = state[env.model.nq:].detach().numpy().copy()
                env.data.ctrl[:] = action.detach().numpy().copy()
                env.data_snapshot = env.get_snapshot()

                next_state = mj_forward()
                env.next_state = next_state

                ctx.data_snapshot = env.data_snapshot
                ctx.reward = env.reward
                ctx.next_state = env.next_state

                return torch.from_numpy(next_state)

            elif mode == "reward":
                ctx.data_snapshot = env.data_snapshot
                ctx.reward = env.reward
                ctx.next_state = env.next_state
                return torch.Tensor([env.reward]).double()

            else:
                raise TypeError("mode has to be 'dynamics' or 'gradient'")

        @staticmethod
        def backward(ctx, grad_output):
            weight = 1 / (max_ep_steps - ctx.data_snapshot.step_idx.value)

            # We should need to calculate gradients only once per dynamics/reward cycle
            if mode == "dynamics":
                state_jacobian = torch.from_numpy(env.dynamics_gradients["state"])
                action_jacobian = torch.from_numpy(env.dynamics_gradients["action"])
                action_jacobian = weight * action_jacobian
            elif mode == "reward":
                # Calculate gradients, "reward" is always called first
                mj_gradients(ctx.data_snapshot, ctx.next_state, ctx.reward)
                state_jacobian = torch.from_numpy(env.reward_gradients["state"])
                action_jacobian = torch.from_numpy(env.reward_gradients["action"])
                action_jacobian = weight * action_jacobian
            else:
                raise TypeError("mode has to be 'dynamics' or 'reward'")
            ds = torch.matmul(grad_output, state_jacobian)
            da = torch.matmul(grad_output, action_jacobian)
            return ds, da

    return MjBlock
