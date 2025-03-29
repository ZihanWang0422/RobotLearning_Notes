import gymnasium as gym 

class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.n_states = env.observation_space.n  # 使用 observation_space.n 替代 nS
        self.n_actions = env.action_space.n      # 使用 action_space.n 替代 nA
        self.v = [0] * self.n_states            # 初始化价值函数
        self.pi = [[0.25] * self.n_actions for _ in range(self.n_states)]  # 初始化均匀随机策略
        self.theta = theta
        self.gamma = gamma

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.v[s]
                self.v[s] = sum(self.pi[s][a] * sum(p * (r + self.gamma * self.v[next_s])
                    for p, next_s, r, _ in self.env.P[s][a]) for a in range(self.n_actions))
                delta = max(delta, abs(v - self.v[s]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.n_states):
            old = self.pi[s].copy()
            q_values = [sum(p * (r + self.gamma * self.v[next_s])
                for p, next_s, r, _ in self.env.P[s][a]) for a in range(self.n_actions)]
            best_a = max(range(self.n_actions), key=lambda a: q_values[a])
            self.pi[s] = [1.0 if a == best_a else 0.0 for a in range(self.n_actions)]
            if old != self.pi[s]:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(4):
        for j in range(4):
            print(f'{agent.v[i*4+j]:6.2f}', end=' ')
        print()

    print("策略：")
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            if state in disaster:
                print('XXXX', end=' ')
            elif state in end:
                print('EEEE', end=' ')
            else:
                policy = agent.pi[state]
                arrows = [action_meaning[a] if prob > 0 else '' for a, prob in enumerate(policy)]
                print(''.join(arrows).ljust(4, 'o'), end=' ')
        print()



env = gym.make("FrozenLake-v1")  # 创建环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])
    
    
# 策略迭代算法
# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

