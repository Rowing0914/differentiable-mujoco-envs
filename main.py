import torch
from differentiable_mujoco.torch_block import launch_state_reward_models
from differentiable_mujoco.launcher import build_env


def main():
    device = "cpu"
    env_name = "InvertedPendulum"
    env_name = "HalfCheetah"
    env_name = "Hopper"
    batch_size = 8
    max_ep_steps = 50

    env = build_env(env_name=env_name, batch_size=batch_size, max_episode_steps=10)
    state_model, reward_model = launch_state_reward_models(env=env, max_ep_steps=10, device=device)
    print(env)
    print(state_model)

    state = torch.tensor(env.reset()).float()
    policy = torch.nn.Linear(env.observation_space.shape[-1], env.action_space.shape[-1])
    value_fn = torch.nn.Linear(env.observation_space.shape[-1] + env.action_space.shape[-1], 1)
    optim = torch.optim.Adam(policy.parameters())
    states, rewards = list(), list()
    for step_idx in range(batch_size):
        action = policy(state.float())
        state = state_model(state, action)
        reward = reward_model(state, action)
        states.append(state)
        rewards.append(reward)
    states = torch.stack(states)
    rewards = torch.stack(rewards)
    actions = policy(states.float())
    Q = value_fn(torch.cat([states.float(), actions.float()], dim=-1))
    loss = -(rewards + Q).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())


if __name__ == "__main__":
    main()
