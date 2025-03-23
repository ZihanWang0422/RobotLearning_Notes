import numpy as np
import matplotlib.pyplot as plt


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()
    
    def createP(self):
        # 初始化
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或终点时无法继续交互
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    
                    # 计算下一个位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    
                    # 判断下一个位置是否在悬崖或终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


# 创建环境实例
env = CliffWalkingEnv(ncol=12, nrow=4)

# 初始化Q表
n_states = env.nrow * env.ncol
n_actions = 4
Q = np.zeros((n_states, n_actions))

# 超参数设置
alpha = 0.1    # 学习率
gamma = 0.99   # 折扣因子
epsilon = 0.1  # 探索率
num_episodes = 500  # 训练的总episode数

# 训练过程
rewards = []  # 记录每个episode的总奖励

for episode in range(num_episodes):
    state = 3 * env.ncol + 0  # 初始状态：左下角(3,0)
    done = False
    total_reward = 0
    
    while not done:
        # epsilon-greedy选择动作
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        
        # 执行动作，得到转移信息
        p, next_state, reward, done = env.P[state][action][0]
        
        # 更新Q表
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
        # 更新状态和总奖励
        state = next_state
        total_reward += reward
        
    rewards.append(total_reward)

# 测试策略
def test_policy(Q):
    state = 3 * env.ncol + 0  # 初始状态
    path = []
    done = False
    while not done:
        action = np.argmax(Q[state])
        path.append((state // env.ncol, state % env.ncol))  # 记录坐标
        p, next_state, reward, done = env.P[state][action][0]
        state = next_state
        if len(path) > 100:  # 防止无限循环
            break
    path.append((state // env.ncol, state % env.ncol))  # 添加终点
    return path

# 可视化结果
path = test_policy(Q)
print("最优路径坐标序列：")
print(path)

# 绘制训练曲线
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Performance')
plt.show()