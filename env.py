import numpy as np

class CliffWalkingEnv:
    """悬崖漫步环境实现"""
    def __init__(self):
        self.nrow = 4    # 网格行数
        self.ncol = 12   # 网格列数
        self.P = self._create_transition_matrix()

    def _create_transition_matrix(self):
        # 定义状态转移矩阵 P[s][a] = [(p, next_state, r, done), ...]
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        actions = [0, 1, 2, 3]  # 上, 下, 左, 右
        for s in range(self.nrow * self.ncol):
            row = s // self.ncol
            col = s % self.ncol
            for a in actions:
                # 默认情况：移动后仍在原地，奖励0，未终止
                transitions = []
                if self._is_cliff(s):
                    # 悬崖状态：立即终止，奖励-100
                    transitions.append((1.0, s, -100, True))
                elif self._is_goal(s):
                    # 终点状态：立即终止，奖励0
                    transitions.append((1.0, s, 0, True))
                else:
                    # 普通状态：计算移动后的位置
                    next_row, next_col = self._move(row, col, a)
                    next_state = next_row * self.ncol + next_col
                    reward = -1  # 每步固定奖励-1
                    done = False
                    # 检查是否移动到悬崖或终点
                    if self._is_cliff(next_state):
                        done = True
                        reward = -100
                    elif self._is_goal(next_state):
                        done = True
                        reward = 0
                    transitions.append((1.0, next_state, reward, done))
                P[s][a] = transitions
        return P

    def _move(self, row, col, action):
        # 根据动作计算新位置
        if action == 0:   # 上
            row = max(row - 1, 0)
        elif action == 1: # 下
            row = min(row + 1, self.nrow - 1)
        elif action == 2: # 左
            col = max(col - 1, 0)
        elif action == 3: # 右
            col = min(col + 1, self.ncol - 1)
        return row, col

    def _is_cliff(self, state):
        # 判断是否为悬崖（状态37到46）
        return (state >= 37) and (state < self.nrow * self.ncol - 1)

    def _is_goal(self, state):
        # 判断是否为终点（状态47）
        return state == self.nrow * self.ncol - 1