import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import flappy_bird_gymnasium

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def play_game(model, num_episodes=5, render_delay=0.1):
    """
    使用训练好的模型玩游戏
    :param model: 训练好的模型
    :param num_episodes: 要展示的游戏回合数
    :param render_delay: 每帧之间的延迟时间（秒）
    """
    env = gym.make('FlappyBird-v0', render_mode='human')
    
    total_score = 0
    best_score = float('-inf')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = model(state_tensor).argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # 添加延迟使游戏更容易观察
            time.sleep(render_delay)
        
        total_score += episode_reward
        best_score = max(best_score, episode_reward)
        print(f"回合 {episode + 1}/{num_episodes}")
        print(f"得分: {episode_reward:.1f}")
        print(f"存活步数: {steps}")
        print("-" * 30)
    
    env.close()
    
    print("\n游戏统计:")
    print(f"平均得分: {total_score/num_episodes:.1f}")
    print(f"最高得分: {best_score:.1f}")

def main():
    # 加载模型
    model = QNetwork().to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'flappy_bird_model.pth')
    
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    print(f"正在加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    
    print("\n开始游戏演示...")
    play_game(model, num_episodes=5, render_delay=0.1)

if __name__ == "__main__":
    main() 