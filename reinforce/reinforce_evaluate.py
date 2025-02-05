
import torch
import pygame
import gymnasium as gym
import torch.nn.functional as F
from torch import nn

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

env_name = "LunarLander-v3"
model_path = "models/reinforce_lunarlander.pth"

env = gym.make(env_name, render_mode="human")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

model = PolicyNet(obs_space, action_space).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

num_episodes_for_eval = 10
reward_history = []

pygame.init()
for episode_idx in range(num_episodes_for_eval):
    observation, info = env.reset()
    episode_reward = 0.0
    done = False

    while not done:
        env.render()
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(state_tensor)
            # 贪心
            action = logits.argmax(dim=1).item()

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    reward_history.append(episode_reward)
    print(f"Episode {episode_idx}, reward: {episode_reward:.2f}")

avg_reward = sum(reward_history) / len(reward_history)
print(f"Average reward over {num_episodes_for_eval} episodes: {avg_reward:.2f}")

env.close()
pygame.quit()
