import torch
import torch.nn as nn
from torch import autograd


def mj_torch_block_factory(env, mode, max_ep_steps, device="cpu"):
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
                env.data.qpos[:] = state[:env.model.nq].detach().cpu().numpy().copy()
                env.data.qvel[:] = state[env.model.nq:].detach().cpu().numpy().copy()
                env.data.ctrl[:] = action.detach().cpu().numpy().copy()
                env.data_snapshot = env.get_snapshot()

                next_state = mj_forward()
                env.next_state = next_state

                ctx.data_snapshot = env.data_snapshot
                ctx.reward = env.reward
                ctx.next_state = env.next_state

                # return torch.from_numpy(next_state)
                return torch.tensor(next_state, device=device)

            elif mode == "reward":
                ctx.data_snapshot = env.data_snapshot
                ctx.reward = env.reward
                ctx.next_state = env.next_state
                # return torch.Tensor([env.reward]).double()
                return torch.tensor([env.reward], device=device).double()

            else:
                raise TypeError("mode has to be 'dynamics' or 'gradient'")

        @staticmethod
        def backward(ctx, grad_output):
            weight = 1 / (max_ep_steps - ctx.data_snapshot.step_idx.value)

            # We should need to calculate gradients only once per dynamics/reward cycle
            if mode == "dynamics":
                # state_jacobian = torch.from_numpy(env.dynamics_gradients["state"])
                # action_jacobian = torch.from_numpy(env.dynamics_gradients["action"])
                state_jacobian = torch.tensor(env.dynamics_gradients["state"], device=device)
                action_jacobian = torch.tensor(env.dynamics_gradients["action"], device=device)
                action_jacobian = weight * action_jacobian
            elif mode == "reward":
                # Calculate gradients, "reward" is always called first
                mj_gradients(ctx.data_snapshot, ctx.next_state, ctx.reward)
                # state_jacobian = torch.from_numpy(env.reward_gradients["state"])
                # action_jacobian = torch.from_numpy(env.reward_gradients["action"])
                state_jacobian = torch.tensor(env.reward_gradients["state"], device=device)
                action_jacobian = torch.tensor(env.reward_gradients["action"], device=device)
                action_jacobian = weight * action_jacobian
            else:
                raise TypeError("mode has to be 'dynamics' or 'reward'")
            ds = torch.matmul(grad_output, state_jacobian)
            da = torch.matmul(grad_output, action_jacobian)
            return ds, da

    return MjBlock


def launch_state_reward_models(env, max_ep_steps, device):
    class Basic(nn.Module):
        def __init__(self, model):
            super(Basic, self).__init__()
            self.model = model

        def forward(self, state, action):
            return self.model(state, action)

    # s_model = Basic(mj_torch_block_factory(env=env, mode="dynamics", max_ep_steps=max_ep_steps).apply).to(device)
    # r_model = Basic(mj_torch_block_factory(env=env, mode="reward", max_ep_steps=max_ep_steps).apply).to(device)
    s_model = Basic(mj_torch_block_factory(env=env, mode="dynamics", max_ep_steps=max_ep_steps, device=device).apply)
    r_model = Basic(mj_torch_block_factory(env=env, mode="reward", max_ep_steps=max_ep_steps, device=device).apply)
    return s_model.train(), r_model.train()
