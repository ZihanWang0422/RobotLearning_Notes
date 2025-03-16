## Reinforcement Learning

[Python3 面向对象 | 菜鸟教程](https://www.runoob.com/python3/python3-class.html)

[蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl/#/)

## Chapter 1 Intro

![alt text](image-1.png)



## Chapter 2 Markov decision process(MDP)

### 2.1马尔科夫过程

1、随机过程：

已知历史信息（S1, ..., St）时下一个时刻状态为St+1的概率表示成$P(S_{t+1}|S_1,\ldots,S_t)$

2、马尔科夫性质：

当且仅当某时刻的状态只取决于上一时刻的状态时$P(S_{t+1}|S_t)=P(S_{t+1}|S_1,\ldots,S_t)$

3、马尔科夫链：<S, P>

n个状态：S = {S1, S2, ... , Sn}

状态转移矩阵P：$\mathcal{P}=\begin{bmatrix}P(s_1|s_1)&\cdots&P(s_n|s_1)\\\vdots&\ddots&\vdots\\P(s_1|s_n)&\cdots&P(s_n|s_n)\end{bmatrix}$  （每一行和为1）



### 2.2马尔科夫奖励过程（MRP）

1、马尔科夫奖励过程：

$\langle\mathcal{S},\mathcal{P},r,\gamma\rangle$

* $\mathcal{S}$是有限状态的集合。

* $\mathcal{P}$是状态转移矩阵。

* $r$是奖励函数，**某个状态$s$的奖励 $r(s)$指转移到该状态时可以获得奖励的期望**。

* $\gamma$是折扣因子 (discount factor), $\gamma$的取值范围为[0,1)。引入折扣因子的理由为远期利益具有一定不确定性，有时我们更希望能够尽快获得一些奖励，所以我们需要对远期利益打一些折扣。接近 1 的γ更关注长期的累计奖励，接近 0 的γ更考虑短期奖励。

  ​



2、回报：

在一个马尔可夫奖励过程中，从第时刻状态开始，直到终止状态时，所有奖励的衰减之和称为**回报**Gt（Return）

$$G_t=R_t+\gamma R_{t+1}+\gamma^2R_{t+2}+\cdots=\sum_{k=0}^\infty\gamma^kR_{t+k}$$

其中**Rt表示在t时刻获得的奖励**

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

（2）所有状态的价值就组成了**价值函数**（value function）**V(s)**，价值函数的输入为某个状态，输出为这个状态的价值。

$$\begin{aligned}V(s)&=\mathbb{E}[G_t|S_t=s]\\&=\mathbb{E}[R_t+\gamma R_{t+1}+\gamma^2R_{t+2}+\ldots|S_t=s]\\&=\mathbb{E}[R_t+\gamma(R_{t+1}+\gamma R_{t+2}+\ldots)|S_t=s]\\&=\mathbb{E}[R_t+\gamma G_{t+1}|S_t=s]\\&=\mathbb{E}[R_t+\gamma V(S_{t+1})|S_t=s]\end{aligned}$$

对最后一式进行分解，其中，即时奖励的期望为奖励函数的输出：$\mathbb{E}[R_t|S_t=s]=r(s)$

剩余部分$\mathbb{E}[\gamma V(S_{t+1})|S_{t}=s]$ 根据从状态s出发的转移概率可以得到；

因此等式等价为**贝尔曼方程**：

✨$V(s)=r(s)+\gamma\sum_{s^{\prime}\in S}p(s^{\prime}|s)V(s^{\prime})$

（3）矩阵形式

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



### 2.3 马尔科夫决策过程（MDP）

1、MDP = MRP + agent's action

2、组成：

$$\langle\mathcal{S},\mathcal{A},P,r,\gamma\rangle$$

$$\begin{aligned}&\mathcal{S}\text{是状态的集合;}\\&\mathcal{A}\text{是动作的集合;}\\&\gamma\text{是折扣因子;}\\&r(s,a)\text{是奖励函数,此时奖励可以同时取决于状态}s\text{和动作}a\text{,在奖励函数只取决于状态}s\text{时,则}\\&\text{退化为}r(s)\mathrm{;}\\&P(s^{\prime}|s,a)\text{是状态转移函数,表示在状态}s\text{执行动作}a\text{之后到达状态}s^{\prime}\text{的概率。}\end{aligned}$$



