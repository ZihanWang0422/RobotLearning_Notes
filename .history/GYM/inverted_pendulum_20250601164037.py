import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建倒立摆环境
env = gym.make("InvertedPendulum-v4") # 不再需要 render_mode="human" 进行训练

# 定义一个简单的策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 对于连续动作空间，输出通常是动作的均值或直接是动作本身，这里简化为直接输出动作
        action = torch.tanh(self.fc3(x)) # 使用 tanh 将输出限制在 [-1, 1] 之间，与环境的动作空间范围匹配
        return action

# 获取状态空间和动作空间的维度
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]

# 创建策略网络和优化器
policy_net = PolicyNetwork(state_space_dim, action_space_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# 训练参数
num_episodes = 100 # 训练的 episode 数量
episode_rewards = []

print("开始训练...")

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 使用策略网络选择动作
        with torch.no_grad(): # 在这里不进行梯度计算，因为我们只是采样动作
            action = policy_net(state_tensor).cpu().numpy().squeeze()
        
        # 将动作缩放到环境的动作范围
        # InvertedPendulum-v4 的动作范围是 [-3, 3]
        scaled_action = np.clip(action * env.action_space.high[0], env.action_space.low[0], env.action_space.high[0])
        
        next_state, reward, terminated, truncated, info = env.step(scaled_action)
        done = terminated or truncated
        total_reward += reward
        
        state = next_state
    
    episode_rewards.append(total_reward)
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

env.close()

print("训练结束。")

# 绘制奖励曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress: Total Reward per Episode")
plt.grid(True)
plt.show()

print("奖励曲线绘制完成。") 