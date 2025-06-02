import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import math
import sys
import os

# 添加当前目录到系统路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RRT import RRT
from RRTStar import RRTStar

def plot_environment(obstacles, start, goal, rand_area):
    """
    绘制环境、起点和终点
    """
    plt.figure(figsize=(10, 10))
    
    # 绘制障碍物
    for obs in obstacles:
        x_obs, y_obs, width, height = obs
        plt.gca().add_patch(Rectangle((x_obs, y_obs), width, height, 
                                     fc='gray', ec='black'))
    
    # 绘制起点和终点
    plt.plot(start[0], start[1], "bs", markersize=8, label="起点")
    plt.plot(goal[0], goal[1], "rs", markersize=8, label="终点")
    
    plt.axis("equal")
    plt.grid(True)
    plt.xlim(rand_area[0] - 1, rand_area[1] + 1)
    plt.ylim(rand_area[2] - 1, rand_area[3] + 1)
    plt.legend()

def calculate_path_length(path):
    """
    计算路径长度
    """
    length = 0
    for i in range(len(path) - 1):
        dx = path[i][0] - path[i+1][0]
        dy = path[i][1] - path[i+1][1]
        length += math.sqrt(dx**2 + dy**2)
    return length

def compare_algorithms(start, goal, obstacles, rand_area, max_iter=200):
    """
    比较RRT和RRT*算法
    """
    # 创建RRT实例
    rrt = RRT(start=start, goal=goal, obstacles=obstacles, rand_area=rand_area,
             expand_dis=5.0, goal_sample_rate=10, max_iter=max_iter)
    
    # 创建RRT*实例
    rrt_star = RRTStar(start=start, goal=goal, obstacles=obstacles, rand_area=rand_area,
                     expand_dis=5.0, goal_sample_rate=10, max_iter=max_iter,
                     connect_circle_dist=20.0)
    
    # 绘制环境
    plt.figure(figsize=(15, 7))
    
    # RRT算法
    plt.subplot(1, 2, 1)
    for obs in obstacles:
        x_obs, y_obs, width, height = obs
        plt.gca().add_patch(Rectangle((x_obs, y_obs), width, height, 
                                     fc='gray', ec='black'))
    plt.plot(start[0], start[1], "bs", markersize=8)
    plt.plot(goal[0], goal[1], "rs", markersize=8)
    plt.title("RRT算法")
    plt.axis("equal")
    plt.grid(True)
    plt.xlim(rand_area[0] - 1, rand_area[1] + 1)
    plt.ylim(rand_area[2] - 1, rand_area[3] + 1)
    
    # 运行RRT算法并计时
    start_time = time.time()
    rrt_path = rrt.planning(animation=False)
    rrt_time = time.time() - start_time
    
    # 绘制RRT树
    for node in rrt.node_list:
        if node.parent is not None:
            parent = rrt.node_list[node.parent]
            plt.plot([node.x, parent.x], [node.y, parent.y], "-g", alpha=0.7)
    
    # 绘制RRT路径
    if rrt_path is not None:
        plt.plot([x for (x, y) in rrt_path], [y for (x, y) in rrt_path], '-r', linewidth=2)
        rrt_length = calculate_path_length(rrt_path)
        plt.title(f"RRT算法 (路径长度: {rrt_length:.2f}, 时间: {rrt_time:.4f}秒)")
    
    # RRT*算法
    plt.subplot(1, 2, 2)
    for obs in obstacles:
        x_obs, y_obs, width, height = obs
        plt.gca().add_patch(Rectangle((x_obs, y_obs), width, height, 
                                     fc='gray', ec='black'))
    plt.plot(start[0], start[1], "bs", markersize=8)
    plt.plot(goal[0], goal[1], "rs", markersize=8)
    plt.title("RRT*算法")
    plt.axis("equal")
    plt.grid(True)
    plt.xlim(rand_area[0] - 1, rand_area[1] + 1)
    plt.ylim(rand_area[2] - 1, rand_area[3] + 1)
    
    # 运行RRT*算法并计时
    start_time = time.time()
    rrt_star_path = rrt_star.planning(animation=False)
    rrt_star_time = time.time() - start_time
    
    # 绘制RRT*树
    for node in rrt_star.node_list:
        if node.parent is not None:
            parent = rrt_star.node_list[node.parent]
            plt.plot([node.x, parent.x], [node.y, parent.y], "-g", alpha=0.7)
    
    # 绘制RRT*路径
    if rrt_star_path is not None:
        plt.plot([x for (x, y) in rrt_star_path], [y for (x, y) in rrt_star_path], '-r', linewidth=2)
        rrt_star_length = calculate_path_length(rrt_star_path)
        plt.title(f"RRT*算法 (路径长度: {rrt_star_length:.2f}, 时间: {rrt_star_time:.4f}秒)")
    
    plt.tight_layout()
    plt.suptitle("RRT vs RRT*算法比较", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # 返回结果
    results = {
        "rrt_path": rrt_path,
        "rrt_time": rrt_time,
        "rrt_length": calculate_path_length(rrt_path) if rrt_path else None,
        "rrt_star_path": rrt_star_path,
        "rrt_star_time": rrt_star_time,
        "rrt_star_length": calculate_path_length(rrt_star_path) if rrt_star_path else None
    }
    
    return results

def create_complex_environment():
    """
    创建一个更复杂的环境进行测试
    """
    # 起点和终点
    start = [10, 10]
    goal = [90, 90]
    
    # 随机采样区域
    rand_area = [0, 100, 0, 100]
    
    # 设置障碍物 [x, y, width, height]
    obstacles = [
        [20, 20, 10, 30],
        [40, 50, 30, 10],
        [60, 30, 10, 30],
        # 减少一些障碍物，留出更多空间
        [30, 70, 30, 10],
        [70, 70, 10, 20],
        [10, 50, 15, 5],
        [50, 10, 5, 15]
    ]
    
    return start, goal, obstacles, rand_area

def main():
    """
    主函数
    """
    print("比较RRT和RRT*算法")
    
    # 创建环境
    start, goal, obstacles, rand_area = create_complex_environment()
    
    # 比较算法，增加最大迭代次数
    results = compare_algorithms(start, goal, obstacles, rand_area, max_iter=2000)
    
    # 打印结果
    print("\n===== 算法比较结果 =====")
    print(f"RRT 算法:")
    print(f"  - 路径长度: {results['rrt_length']:.2f}" if results['rrt_length'] else "  - 未找到路径")
    print(f"  - 计算时间: {results['rrt_time']:.4f}秒")
    
    print(f"\nRRT* 算法:")
    print(f"  - 路径长度: {results['rrt_star_length']:.2f}" if results['rrt_star_length'] else "  - 未找到路径")
    print(f"  - 计算时间: {results['rrt_star_time']:.4f}秒")
    
    if results['rrt_length'] and results['rrt_star_length']:
        improvement = (results['rrt_length'] - results['rrt_star_length']) / results['rrt_length'] * 100
        print(f"\nRRT*相比RRT路径长度改进: {improvement:.2f}%")
    
if __name__ == "__main__":
    main() 