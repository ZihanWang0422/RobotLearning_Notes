import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import deque
import random
import flappy_bird_gymnasium
import time
import os

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make('FlappyBird-v0', render_mode='human')

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(180, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train(model, target_model, env, num_episodes=1000, gamma=0.99, lr=0.0001, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # ε-greedy策略
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    action = model(state_tensor).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            
            if len(replay_buffer) >= batch_size:
                # 从经验回放中采样
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # 将数据转换为张量并移到GPU
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # 计算目标Q值
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                # 计算当前Q值
                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # 计算损失并更新
                loss = F.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
        
        # 更新目标网络
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

def visualize_game(model, num_episodes=5, render_delay=0.1):
    """
    可视化训练好的模型玩游戏
    :param model: 训练好的模型
    :param num_episodes: 要展示的游戏回合数
    :param render_delay: 每帧之间的延迟时间（秒）
    """
    env = gym.make('FlappyBird-v0', render_mode='human')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = model(state_tensor).argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 添加延迟使游戏更容易观察
            time.sleep(render_delay)
        
        print(f"Episode {episode + 1}, Score: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    # 创建主网络和目标网络
    model = QNetwork().to(device)
    target_model = QNetwork().to(device)
    target_model.load_state_dict(model.state_dict())

    # 训练模型
    print("开始训练模型...")
    train(model, target_model, env)
    
    # 保存训练好的模型
    model_path = os.path.join(os.path.dirname(__file__), 'flappy_bird_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    # 可视化训练好的模型
    print("\n开始可视化游戏...")
    visualize_game(model, num_episodes=5, render_delay=0.1)

# print([env for env in gym.envs.registry])