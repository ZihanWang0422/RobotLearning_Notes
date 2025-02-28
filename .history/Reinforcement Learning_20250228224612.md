## Reinforcement Learning

[Python3 面向对象 | 菜鸟教程](https://www.runoob.com/python3/python3-class.html)



## Chapter 1 Intro

### 1.1



## Chapter 2 Markov decision process(MDP)

### 2.1马尔科夫过程

1、随机过程：

已知历史信息（S1, ..., St）时下一个时刻状态为St+1的概率表示成$P(S_{t+1}|S_1,\ldots,S_t)$

2、马尔科夫性质：

当且仅当某时刻的状态只取决于上一时刻的状态时$P(S_{t+1}|S_t)=P(S_{t+1}|S_1,\ldots,S_t)$

3、马尔科夫链：<S, P>

n个状态：S = {S1, S2, ... , Sn}

状态转移矩阵P：$\mathcal{P}=\begin{bmatrix}P(s_1|s_1)&\cdots&P(s_n|s_1)\\\vdots&\ddots&\vdots\\P(s_1|s_n)&\cdots&P(s_n|s_n)\end{bmatrix}$  （每一行和为1）



### 2.2马尔科夫奖励过程

1、马尔科夫奖励过程：

$\langle\mathcal{S},\mathcal{P},r,\gamma\rangle$

* $\mathcal{S}$是有限状态的集合。
* $\mathcal{P}$是状态转移矩阵。
*  $r$是奖励函数，某个状态$s$的奖励 $r(s)$指转移到该状态时可以获得奖励的期望。
*  $\gamma$是折扣因子 (discount factor), $\gamma$的取值范围为[0,1)。引入折扣因子的理由为远期利益具有一定不确定性，有时我们更希望能够尽快获得一些奖励，所以我们需要对远期利益打一些折扣。接近 1 的γ更关注长期的累计奖励，接近 0 的γ更考虑短期奖励。



2、回报：

在一个马尔可夫奖励过程中，从第时刻状态开始，直到终止状态时，所有奖励的衰减之和称为**回报**Gt（Return）



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
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1] #使用reversed先算chain最后一位，每一次都会对其进行乘gamma
    return G


# 一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为：%s。" % G)
```

