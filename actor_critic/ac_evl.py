import pygame
import gymnasium as gym
import torch
import numpy as np
from torch import nn
import sklearn
from sklearn.preprocessing import StandardScaler

env = gym.make("MountainCarContinuous-v0", render_mode="human")

def scale_state(state, scaler):
    state = np.array(state, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(state)
    return torch.tensor(scaled, dtype=torch.float32)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 40)
        self.fc2 = nn.Linear(40, 40)
        self.mu = nn.Linear(40, action_dim)
        self.sigma = nn.Linear(40, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = torch.nn.functional.softplus(self.sigma(x)) + 0.1
        return mu, sigma

scaler = sklearn.preprocessing.StandardScaler()
state_space_samples = np.array([env.observation_space.sample() for _ in range(10000)])
scaler.fit(state_space_samples)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
policy_net = PolicyNet(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load("models/actor_critic_mountaincar.pth", map_location=device))
policy_net.eval()

def evaluate_policy(env, policy_net, scaler, num_episodes=10):
    reward_history = []

    for episode_idx in range(num_episodes):
        state, _ = env.reset(seed=episode_idx)
        state = scale_state(state, scaler).to(device)
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                mu, sigma = policy_net(state)
                dist = torch.distributions.Normal(mu, sigma)
                action = dist.sample()  # 从正态分布采样动作
                action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            state = scale_state(next_state, scaler).to(device)
            episode_reward += reward
            done = terminated or truncated

        reward_history.append(episode_reward)
        print(f"Episode {episode_idx + 1}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(reward_history)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

evaluate_policy(env, policy_net, scaler, num_episodes=10)

env.close()
