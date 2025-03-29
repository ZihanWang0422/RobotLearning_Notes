import copy

class PolicyIteration:
    """策略迭代算法"""
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * (self.env.ncol * self.env.nrow)  # 初始化价值函数
        # 初始化均匀随机策略（每个动作概率相等）
        self.pi = [[0.25] * 4 for _ in range(self.env.ncol * self.env.nrow)]
        self.theta = theta    # 策略评估收敛阈值
        self.gamma = gamma    # 折扣因子

    def compute_q_values(self, s):
        """计算指定状态下所有动作的Q值"""
        q_values = []
        for a in range(4):
            q = 0
            for trans_prob, next_state, reward, done in self.env.P[s][a]:
                # 计算Q值：考虑转移概率和终止状态处理
                q += trans_prob * (reward + self.gamma * self.v[next_state] * (1 - done))
            q_values.append(q)
        return q_values

    def policy_evaluation(self):
        """策略评估（使用当前策略更新价值函数）"""
        iteration = 0
        while True:
            max_diff = 0
            new_v = [0] * len(self.v)
            for s in range(len(self.v)):
                # 使用当前策略计算状态价值
                q_values = self.compute_q_values(s)
                new_v[s] = sum(self.pi[s][a] * q_values[a] for a in range(4))
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            
            self.v = new_v
            iteration += 1
            
            if max_diff < self.theta:
                print(f"策略评估经过{iteration}轮迭代后收敛")
                return

    def policy_improvement(self):
        """策略提升（根据当前价值函数优化策略）"""
        policy_stable = True
        for s in range(len(self.v)):
            # 获取当前状态的所有Q值
            q_values = self.compute_q_values(s)
            
            # 找到最大Q值对应的动作
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            
            # 构造新的策略（均匀分布在最优动作上）
            new_policy = [0.] * 4
            prob = 1.0 / len(best_actions)
            for a in best_actions:
                new_policy[a] = prob
                
            # 检查策略是否变化
            if self.pi[s] != new_policy:
                policy_stable = False
                
            self.pi[s] = new_policy
            
        return policy_stable

    def policy_iteration(self):
        """执行完整的策略迭代"""
        iteration = 0
        while True:
            iteration += 1
            print(f"\n=== 策略迭代第{iteration}轮 ===")
            
            self.policy_evaluation()
            # 进行策略提升并检查策略稳定性
            if self.policy_improvement():
                print("策略已稳定，停止迭代")
                return

def print_agent(agent, action_meaning, disaster=[], end=[]):
    """可视化状态价值和策略"""
    print("状态价值：")
    grid_size = agent.env.ncol
    for i in range(agent.env.nrow):
        for j in range(grid_size):
            state = i * grid_size + j
            print(f"{agent.v[state]:6.3f}", end=" ")
        print()

    print("\n策略：")
    for i in range(agent.env.nrow):
        for j in range(grid_size):
            state = i * grid_size + j
            if state in disaster:
                print("XXXX", end=" ")
            elif state in end:
                print("EEEE", end=" ")
            else:
                policy = agent.pi[state]
                # 将策略转换为箭头表示
                arrows = [action_meaning[a] if prob > 0 else "" for a, prob in enumerate(policy)]
                combined = "".join(arrows).ljust(4, 'o')  # 保持4字符宽度
                print(combined, end=" ")
        print()

# 示例使用（需要CliffWalkingEnv环境实现）
if __name__ == "__main__":
    from env import CliffWalkingEnv  # 假设环境已实现
    
    env = CliffWalkingEnv()
    action_meaning = ['↑', '↓', '←', '→']  # 上，下，左，右
    
    # 初始化策略迭代算法参数
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    
    # 执行策略迭代
    agent.policy_iteration()
    
    # 打印结果（悬崖状态37-46，终点47）
    print_agent(agent, action_meaning, list(range(37, 47)), [47])