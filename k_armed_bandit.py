#coding:utf8
'''
最简单的例子K-armed bandit problem
@author hy@ttt

'''
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.DEBUG)
class Bandit:

    # 首先定义bandit
    # 每个bandit的初始化,定义10个动作，以及对应是个动作的价值从N(o,1)的正态分布中随机得到
    # 保证每一个bandit的真正value函数不一样
    def __init__(self,k):
        self._actions = range(k)
        self._values = [i for i in np.random.normal(0, 1, k)]
        logging.info('初始化bandit,{:}个动作 '.format(k))


    # 定义一个函数，返回做出该动作的reward
    def reward(self,a):
        # reward 是从action a 的真实价值 self._values[a] 为 均值，1为方差的正态分布中获得的
        mu = self._values[a]
        reward = np.random.normal(mu,1,1)[0]
        return reward

    # 每一个bandit 有自己的最优的action
    def optimal_action(self):
        return self._values.index(max(self._values))


# 定义一个赌徒，也就是agent
class Gambler:

    # 初始化赌徒的经验值
    def __init__(self,k=10,_epsilon=0,value_func_name='action_value'):
        # 初始化epsilon
        self._epsilon =  _epsilon
        # 初始化对每个价值的判断字典
        self._actions = range(k)
        self._value_func_name = value_func_name
        if self._value_func_name =='action_value':
            self._value_func = self.action_value
            self._action_values = defaultdict(list)

        elif self._value_func_name == 'incremental':
            self._n = defaultdict(int)
            self._Q = defaultdict(int)
            self._value_func = self.incremental_impl
            

        logging.info('初始化gambler,{:}个动作,epsilon:{:},value func:{:}'.format(k,epsilon,value_func_name))


    # 根据当前已知的判断 选择出最优的 action
    def argmax_a_Qa(self):
        # 最简单的就是遍历一遍价值词典，计算每个动作的价值
        action_values = [self._value_func(a) for a in self._actions]
        max_value = max(action_values)

        # 获得最大值所在的所有位置
        actions = [i for i,v in enumerate(action_values) if v==max_value]

        #从这些最大值动作里随机选择一个最为决策
        return np.random.choice(actions)

    # 定义action value的函数，也就是一个动作的价值等于之前出现获得的reward的平均值
    def action_value(self,a):
        reward_list = self._action_values.get(a,[])
        if len(reward_list)==0:
            #当该value不存时，对应的值是0
            Q_a = 0
        else:
            # 每个action的估计Value是其之前所有返回reward的均值
            Q_a = sum(reward_list)/float(len(reward_list))

        return Q_a

    # 在上一步会计算完成下一步需要的value,直接获取，要么为0
    def incremental_impl(self,a):
        return self._Q.get(a,0)


    def update_action_values(self,a,reward):
        if self._value_func_name=='action_value':
            self._action_values[a].append(reward)
        elif self._value_func_name=='incremental':
            self._n[a]+=1 
            self._Q[a] = self._Q[a]+(1/self._n[a])*(reward-self._Q[a])
            


    # 执行动作最终由epsilon的随机数决定的
    def epsilon_decision(self):
        # 获得最优的action a
        a = self.argmax_a_Qa()

        # 获得一个[0,1)随机数
        rni = np.random.random()

        # 如果随机数小于 epsilon
        if rni < self._epsilon:
            # 从其他的备选集中选择
            a = np.random.choice([ac for ac in self._actions if ac!=a])

        # 返回最优动作a
        return a


def k_armed_bandit_run(n_loops=1000,epsilon=0,k=10,value_func_name='incremental'):
    # 两个数组记录每一个循环的reward和最优动作
    rewards = []
    isoptimals = []
    #初始化一个赌徒
    gambler = Gambler(_epsilon=epsilon,k=k,value_func_name=value_func_name)
    #初始化一个bandit
    bandit =  Bandit(k=k)
    #最优动作是
    optimal = bandit.optimal_action()
    #这个赌徒做n_loops次选择
    for _ in range(n_loops):
        #每一次 首先获得决策最优
        action = gambler.epsilon_decision()
        # 获得这个动作的reward
        reward = bandit.reward(action)
        # 赌徒更新价值函数
        gambler.update_action_values(action,reward)

        # 记录这个动作的reward
        rewards.append(reward)
        #记录这个动作是否最优
        isoptimals.append(1 if action==optimal else 0)

    #返回记录的数组
    return rewards,[sum(isoptimals[:i+1])/float(len(isoptimals[:i+1])) for i,o in enumerate(isoptimals)]

# 多次实验的结果
def N_runs(N=2000,epsilon=0,n_loops=1000,k=10,value_func_name='incremental'):

    # 记录每一个循环的结果值
    runs_rewards = []
    runs_optimals = []

    for i in range(N):
        logging.info('--- Epsilon:{:},Round:{:} ---'.format(epsilon,i))
        rewards,optimals = k_armed_bandit_run(n_loops=n_loops,epsilon=epsilon,k=k,value_func_name=value_func_name)
        runs_optimals.append(optimals)
        runs_rewards.append(rewards)

    return runs_rewards,runs_optimals

# 画图
def plot_N_runs(ax1,ax2,_N_rewards,_N_optimals,epsilon,n_loops):
    
    xs = np.array(range(n_loops))+1
    # plot reward
    rewards = np.array(_N_rewards).mean(axis=0)
    ax1.plot(xs,rewards,label='$\epsilon$:{:}'.format(epsilon))

    #plot optimal
    optimals = np.array(_N_optimals).mean(axis=0)
    ax2.plot(xs,optimals,label='$\epsilon$:{:}'.format(epsilon))

    ax1.set_title("Average Rewards Trends")
    ax1.set_xlabel("Average Rewards")
    ax1.set_ylabel("time step t")

    ax2.set_title("Optimal Rate along time step t")
    ax2.set_xlabel("Optimal Rate")
    ax2.set_ylabel("time step t")


if __name__ == '__main__':
    
    # 定义N 
    N =200
    n_loops = 1000
    k=10
    epsilons = [0,0.01,0.1]
    # 定义value function的计算方式
    value_func_name = 'incremental'
    # 在完成两千个循环之后
    fig,axes = plt.subplots(2,1,figsize=(10,10))
    ax1 = axes[0]
    ax2 = axes[1]
    for epsilon in epsilons:
        runs_rewards,runs_optimals = N_runs(N=N,epsilon=epsilon,n_loops=n_loops,k=k,value_func_name=value_func_name)
        plot_N_runs(ax1,ax2,runs_rewards,runs_optimals,epsilon,n_loops)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig('{:}-armed-bandit-{:}-{:}-{:}.png'.format(k,value_func_name,N,n_loops))

















