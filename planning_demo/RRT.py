import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
import math

class Node:
    """
    RRT树的节点类
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    """
    RRT算法实现类
    """
    def __init__(self, start, goal, obstacles, rand_area, 
                 expand_dis=3.0, goal_sample_rate=5, max_iter=500):
        """
        初始化RRT
        
        参数:
            start: 起点坐标 [x, y]
            goal: 终点坐标 [x, y]
            obstacles: 障碍物列表，每个障碍物为 [x, y, width, height]
            rand_area: 随机采样区域 [min_x, max_x, min_y, max_y]
            expand_dis: 扩展距离
            goal_sample_rate: 采样目标点的概率 (%)
            max_iter: 最大迭代次数
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y = rand_area
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        
    def planning(self, animation=True):
        """
        RRT路径规划
        
        参数:
            animation: 是否显示动画
            
        返回:
            path: 规划的路径，节点列表
        """
        for i in range(self.max_iter):
            # 生成随机节点
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd = self.get_random_node()
            else:
                rnd = Node(self.goal.x, self.goal.y)
                
            # 找到最近的节点
            nearest_ind = self.get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # 从最近节点向随机节点扩展
            new_node = self.steer(nearest_node, rnd, self.expand_dis)
            new_node.parent = nearest_ind
            
            # 检查是否碰撞
            if not self.check_collision(new_node, self.obstacles):
                self.node_list.append(new_node)
                
                # 显示动画
                if animation:
                    self.draw_graph(new_node)
                
                # 检查是否到达目标
                if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                    final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                    final_node.parent = len(self.node_list) - 1
                    
                    if not self.check_collision(final_node, self.obstacles):
                        return self.generate_final_path(len(self.node_list) - 1)
                        
        return None  # 达到最大迭代次数仍未找到路径
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        从from_node向to_node方向扩展一个新节点
        """
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        
        if extend_length > d:
            extend_length = d
            
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        
        return new_node
    
    def generate_final_path(self, goal_ind):
        """
        从目标节点回溯生成路径
        """
        path = [[self.goal.x, self.goal.y]]
        node_ind = goal_ind
        while node_ind is not None:
            node = self.node_list[node_ind]
            path.append([node.x, node.y])
            node_ind = node.parent
            
        return path
    
    def calc_dist_to_goal(self, x, y):
        """
        计算到目标点的距离
        """
        return math.hypot(x - self.goal.x, y - self.goal.y)
    
    def get_random_node(self):
        """
        在指定区域内随机生成节点
        """
        x = random.uniform(self.min_rand_x, self.max_rand_x)
        y = random.uniform(self.min_rand_y, self.max_rand_y)
        node = Node(x, y)
        return node
    
    def get_nearest_node_index(self, rnd_node):
        """
        找到离随机节点最近的树中节点的索引
        """
        dlist = [(node.x - rnd_node.x) ** 2 + 
                 (node.y - rnd_node.y) ** 2 for node in self.node_list]
        minind = dlist.index(min(dlist))
        return minind
    
    def check_collision(self, node, obstacles):
        """
        检查节点是否与障碍物碰撞
        """
        if node is None:
            return False
        
        for obs in obstacles:
            x_obs, y_obs, width, height = obs
            
            if (node.x > x_obs - 0.5 and node.x < x_obs + width + 0.5 and
                node.y > y_obs - 0.5 and node.y < y_obs + height + 0.5):
                return True  # 碰撞
        
        return False  # 无碰撞
    
    def calc_distance_and_angle(self, from_node, to_node):
        """
        计算两个节点之间的距离和角度
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    def draw_graph(self, rnd=None):
        """
        绘制当前RRT树和环境
        """
        plt.clf()
        
        # 绘制障碍物
        for obs in self.obstacles:
            x_obs, y_obs, width, height = obs
            plt.gca().add_patch(Rectangle((x_obs, y_obs), width, height, 
                                         fc='gray', ec='black'))
        
        # 绘制起点和终点
        plt.plot(self.start.x, self.start.y, "bs", markersize=8)
        plt.plot(self.goal.x, self.goal.y, "rs", markersize=8)
        
        # 绘制RRT树
        for node in self.node_list:
            if node.parent is not None:
                parent = self.node_list[node.parent]
                plt.plot([node.x, parent.x], [node.y, parent.y], "-g")
        
        # 绘制随机采样点
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            
        plt.axis("equal")
        plt.grid(True)
        plt.xlim(self.min_rand_x - 1, self.max_rand_x + 1)
        plt.ylim(self.min_rand_y - 1, self.max_rand_y + 1)
        plt.title("RRT算法演示")
        plt.pause(0.01)
        
    def draw_final_path(self, path):
        """
        绘制最终路径
        """
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', linewidth=2)
        plt.pause(0.01)


def main():
    """
    主函数，演示RRT算法
    """
    print("RRT路径规划算法演示")
    
    # 设置起点和终点
    start = [5, 5]
    goal = [45, 45]
    
    # 设置随机采样区域
    rand_area = [0, 50, 0, 50]
    
    # 设置障碍物 [x, y, width, height]
    obstacles = [
        [10, 10, 10, 10],
        [20, 30, 10, 5],
        [30, 15, 5, 15],
        [40, 30, 5, 5]
    ]
    
    # 创建RRT实例
    rrt = RRT(start=start, goal=goal, obstacles=obstacles, rand_area=rand_area,
             expand_dis=3.0, goal_sample_rate=5, max_iter=500)
    
    # 规划路径
    path = rrt.planning(animation=True)
    
    if path is None:
        print("未找到路径!")
    else:
        print("找到路径!")
        
        # 绘制最终路径
        rrt.draw_final_path(path)
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
