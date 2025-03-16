## Reinforcement Learning

[Python3 面向对象 | 菜鸟教程](https://www.runoob.com/python3/python3-class.html)

[蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl/#/)

## Chapter 1 Intro

![alt text](image-1.png)



1.1奖励

由环境给的一种标量的反馈信号（scalar feedback signal），这种信号可显示智能体在某一步采取某个策略的表现如何。强化学习的目的就是最大化智能体可以获得的奖励，智能体在环境里面存在的目 的就是最大化它的期望的累积奖励（expected cumulative reward）。



1.2 智能体组成部分

- **策略（policy）**：将输入的状态变成可能采取的动作的概率，通常采用随机策略，在学习时可以通过引入一定的随机性来更好地探索环境。

**随机性策略（π函数）**：$\pi(a|s)=p\left(a_{t}=a|s_{t}=s\right)$输入状态s，输出智能体所有动作得概率。

**确定性策略**： 智能体直接采取最有可能的动作，$a^*=\arg\max_a\pi(a\mid s)$



- **价值函数（value function）**。我们用价值函数来对当前状态进行评估。价值函数用于评估智能体进 入某个状态后，可以对后面的奖励带来多大的影响。价值函数值越大，说明智能体进入这个状态越有利。

  ​

- **模型（model）**。模型表示智能体对环境的状态进行理解，它决定了环境中世界的运行方式。 

  ​



1.3 智能体类型

* **基于价值的智能体**：显式地学习价值函数，隐式地学习它的策略。策略是其从学到的价值函数里面推算出来的（Q-learning/Sarsa）



* **基于策略的智能体**：直接学习策略，我们给它一个状态，它就会输出对应动作的概率。（Policy Gradient/PG）

  ​

```python
import gymnasium as gym

#创建一个env 参数包括场景类型、渲染模式
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

#首先设置eposode_over为flase，进入训练循环
episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action) #智能体执行动作，智能体从更新后的环境接收到新的观测以及执行动作的奖励；

    episode_over = terminated or truncated

env.close()
```



## Chapter 2 Markov decision process(MDP)

### 2.1马尔科夫过程

1、随机过程：

已知历史信息（S1, ..., St）时下一个时刻状态为St+1的概率表示成$P(S_{t+1}|S_1,\ldots,S_t)$

2、马尔科夫性质：（**状态转移**）

当且仅当某时刻的状态只取决于上一时刻的状态时$P(S_{t+1}|S_t)=P(S_{t+1}|S_1,\ldots,S_t)$

3、马尔科夫链：<S, P>

n个状态：S = {S1, S2, ... , Sn}

状态转移矩阵P：$\mathcal{P}=\begin{bmatrix}P(s_1|s_1)&\cdots&P(s_n|s_1)\\\vdots&\ddots&\vdots\\P(s_1|s_n)&\cdots&P(s_n|s_n)\end{bmatrix}$  （每一行和为1）



### 2.2马尔科夫奖励过程（MRP）

1、马尔科夫奖励过程：

$\langle\mathcal{s},\mathcal{P},r,\gamma\rangle$

* $\mathcal{s}$是有限状态的集合。

* $\mathcal{P}$是状态转移矩阵。

* $r$是奖励函数，**某个状态$s$的奖励 $r(s)$指转移到该状态时可以获得奖励的期望**。

* $\gamma$是折扣因子 (discount factor), $\gamma$的取值范围为[0,1)。引入折扣因子的理由为远期利益具有一定不确定性，有时我们更希望能够尽快获得一些奖励，所以我们需要对远期利益打一些折扣。接近 1 的γ更关注长期的累计奖励，接近 0 的γ更考虑短期奖励。

  ​



2、回报：

在一个马尔可夫奖励过程中，从第t时刻状态开始，直到终止状态时，所有奖励的衰减之和称为**回报**Gt（Return）

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots=\sum_{k=0}^\infty\gamma^kr_{t+k}$$

其中**r_t表示在t时刻获得的奖励**

```python 
import numpy as np
np.random.seed(0)
# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数
gamma = 0.5  # 定义折扣因子


# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma):
    G = 0
    for i in range(len(chain)-1, -1, len(chain)):
        G = gamma * G + rewards[chain[i] - 1] #使用reversed先算chain最后一位，每一次都会对其进行乘gamma
    return G


# 一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为：%s。" % G)
```



3、价值函数：

（1）马尔可夫奖励过程中，一个状态的期望回报（即从这个状态出发的未来累积奖励的期望）被称为这个状态的**价值**（value）。

🙌注意价值是针对某个起始状态t的!!!

（2）所有状态的价值就组成了**价值函数**（value function）**V(s)**，价值函数的输入为某个状态，输出为这个状态的价值。

$$\begin{aligned}V(s)&=\mathbb{E}[G_t|s_t=s]\\&=\mathbb{E}[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\ldots|s_t=s]\\&=\mathbb{E}[r_t+\gamma(r_{t+1}+\gamma r_{t+2}+\ldots)|s_t=s]\\&=\mathbb{E}[r_t+\gamma G_{t+1}|s_t=s]\\&=\mathbb{E}[r_t+\gamma V(s_{t+1})|s_t=s]\end{aligned}$$

对最后一式进行分解，其中，即时奖励的期望为奖励函数的输出：$\mathbb{E}[r_t|s_t=s]=R(s)$

剩余部分$\mathbb{E}[\gamma V(s_{t+1})|s_{t}=s]$ 根据从状态s出发的转移概率可以得到；

因此等式等价为**贝尔曼方程**：

✨$V(s)=R(s)+\gamma\sum_{s^{\prime}\in S}p(s^{\prime}|s)V(s^{\prime})$

**（3）矩阵形式**

$$\mathcal{V}=\mathcal{R}+\gamma\mathcal{PV}\\\begin{bmatrix}V(s_1)\\V(s_2)\\\cdots\\V(s_n)\end{bmatrix}=\begin{bmatrix}r(s_1)\\r(s_2)\\\cdots\\r(s_n)\end{bmatrix}+\gamma\begin{bmatrix}P(s_1|s_1)&p(s_2|s_1)&\ldots&P(s_n|s_1)\\P(s_1|s_2)&P(s_2|s_2)&\ldots&P(s_n|s_2)\\\cdots\\P(s_1|s_n)&P(s_2|s_n)&\ldots&P(s_n|s_n)\end{bmatrix}\begin{bmatrix}V(s_1)\\V(s_2)\\\ldots\\V(s_n)\end{bmatrix}$$

解析解：$$\mathcal{V}=(I-\gamma\mathcal{P})^{-1}\mathcal{R}$$ 时间复杂度为O(n^3)

```python
def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


V = compute(P, rewards, gamma, 6)
print("MRP中每个状态价值分别为\n", V)
```



4、迭代算法：

（1）蒙特卡洛采样算法：(相当于对回报求均值)

当得到一个马尔可夫奖励过程后，我们可以从某个状态开始，把小船放到状态转移矩阵里面，让它“随波逐流”，这样就会产生一个轨迹。产生一个轨迹之后，就会得到一个奖励，那么直接把折扣的奖励即回报 g 算出来。算出来之后将它积累起来，得到回报Gt。 当积累了一定数量的轨迹之后，我们直接用 Gt除以轨迹数量，就会得到某个状态的价值

![alt text](image-2.png)



（2）动态规划算法：

通过**自举（bootstrapping）**的方法不停地迭代贝尔曼方程，当最后更新的状态与我们上一个状态的区别并不大的时候，更新就可以停止，我们就可以输出最新的 V′(s) 作为它当前的状态的价值。

![alt text](image-3.png)



### 2.3 马尔科夫决策过程（MDP）

1、组成：MDP = MRP + agent's action
![alt text](image-4.png)

$$\langle\mathcal{s},\mathcal{a},P,r,\gamma\rangle$$

$$\begin{aligned}&\mathcal{s}\text{是状态的集合;}\\&\mathcal{a}\text{是动作的集合;}\\&\gamma\text{是折扣因子;}\\&r(s,a)\text{是奖励函数,此时奖励可以同时取决于状态}s\text{和动作}a\text{,在奖励函数只取决于状态}s\text{时,则}\\&\text{退化为}r(s)\mathrm{;}\\&P(s^{\prime}|s,a)\text{是状态转移函数,表示在状态}s\text{执行动作}a\text{之后到达状态}s^{\prime}\text{的概率。}\end{aligned}$$



2、状态转移：

状态转移：$p\left(s_{t+1}=s^{\prime}\mid s_t=s,a_t=a\right)$

MDP满足条件：$p\left(s_{t+1}\mid s_t,a_t\right)=p\left(s_{t+1}\mid h_t,a_t\right)$



3、策略：

（1）状态、动作概率：

在某个状态可能采取某个行动的概率 

$\pi(a\mid s)=p\left(a_t=a\mid s_t=s\right)$



（2）策略转化：

对动作进行加权，得到MRP的**状态转移概率**

 $P_\pi\left(s^{\prime}\mid s\right)=\sum_{a\in A}\pi(a\mid s)p\left(s^{\prime}\mid s,a\right)$



4、价值函数：