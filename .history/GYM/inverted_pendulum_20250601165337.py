import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建倒立摆环境
env = gym.make("InvertedPendulum-v4") # 不再需要 render_mode="human" 进行训练

# 定义一个策略网络 - 输出动作的均值和标准差
class PolicyNetwork(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_space_dim)  # 输出动作均值
        self.log_std = nn.Linear(64, action_space_dim)  # 输出动作标准差的对数
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * env.action_space.high[0]  # 缩放到动作空间
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mean, log_std
    
    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        return action, log_prob
    
    def get_deterministic_action(self, state):
        mean, _ = self.forward(state)
        return mean

# 获取状态空间和动作空间的维度
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]

# 创建策略网络和优化器
policy_net = PolicyNetwork(state_space_dim, action_space_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# 训练参数
num_episodes = 500  # 增加训练的episode数量
gamma = 0.99  # 折扣因子
episode_rewards = []
best_reward = -float('inf')  # 记录最佳奖励

print("开始训练...")

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    log_probs = []
    rewards = []
    
    # 收集一个episode的轨迹
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 使用策略网络选择动作并获取动作的对数概率
        action, log_prob = policy_net.get_action(state_tensor)
        action = action.cpu().detach().numpy().squeeze()
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存储轨迹
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        
        state = next_state
    
    episode_rewards.append(total_reward)
    
    # 计算折扣回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns).to(device)
    
    # 标准化回报以减少方差
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # 计算策略梯度损失
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)  # 负号是因为我们要最大化回报
    
    # 优化策略网络
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
    # 打印训练进度
    if (episode + 1) % 10 == 0:
        avg_reward = sum(episode_rewards[-10:]) / 10
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Avg Reward (last 10) = {avg_reward:.2f}")
    
    # 保存表现最好的模型
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), 'inverted_pendulum_model.pth')
        print(f"保存新的最佳模型，奖励: {best_reward:.2f}")

env.close()

print("训练结束。最佳奖励: {:.2f}".format(best_reward))

# 绘制奖励曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress: Total Reward per Episode")
plt.grid(True)
plt.savefig('training_curve.png')  # 保存图表
plt.show()

print("奖励曲线绘制完成并保存。")

# 测试训练好的模型
print("\n开始测试模型...")
env = gym.make("InvertedPendulum-v4", render_mode="human")  # 创建可视化环境
policy_net.load_state_dict(torch.load('inverted_pendulum_model.pth'))
policy_net.eval()

for _ in range(3):  # 测试3个episode
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 使用确定性策略（取均值）
        with torch.no_grad():
            action = policy_net.get_deterministic_action(state_tensor).cpu().numpy().squeeze()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        state = next_state
        time.sleep(0.01)  # 减慢可视化速度
    
    print(f"测试 Episode: Total Reward = {total_reward:.2f}")

env.close()
print("测试完成。")