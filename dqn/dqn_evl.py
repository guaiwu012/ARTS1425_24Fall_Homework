import torch
import torch.nn.functional as F
from torch import nn
import gymnasium as gym
import numpy as np
from collections import Counter

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def one_hot(state, state_dim):
    vec = torch.zeros(state_dim, dtype=torch.float32)
    vec[state] = 1.0
    return vec

env_name = "Taxi-v3"
model_path = "models/dqn_taxi_v3.pth"

env = gym.make(env_name, render_mode="human")  
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

obs_space = env.observation_space.n
action_space = env.action_space.n

model = QNet(obs_space, action_space).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

num_episodes_for_eval = 10
reward_history = []
action_counts = Counter()  
termination_reasons = {"success": 0, "fail": 0}  # 记录成功/失败次数

for episode_idx in range(num_episodes_for_eval):
    observation, info = env.reset(seed=episode_idx)
    state_vec = one_hot(observation, obs_space).to(device) 
    episode_reward = 0.0
    done = False
    
    while not done:
        env.render()  
        with torch.no_grad():
            q_values = model(state_vec.unsqueeze(0))  
            action = q_values.argmax(dim=1).item()   
            action_counts[action] += 1  

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state_vec = one_hot(next_observation, obs_space).to(device)

        state_vec = next_state_vec  # 更新状态
        episode_reward += reward
        done = terminated or truncated  # 检查结束

        # 终止原因
        if done:
            if reward == 20:  
                termination_reasons["success"] += 1
            else:
                termination_reasons["fail"] += 1

    reward_history.append(episode_reward)
    print(f"Episode {episode_idx}, reward: {episode_reward:.2f}")

# 平均奖励
avg_reward = sum(reward_history) / len(reward_history)
print(f"\nAverage reward over {num_episodes_for_eval} episodes: {avg_reward:.2f}")
print(f"Action counts: {action_counts}")
print(f"Termination reasons: {termination_reasons}")

env.close()
