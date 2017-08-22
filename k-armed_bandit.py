#coding:utf-8
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
'''
最简单的例子K-armed bandit problem

'''

## 首先定义一些变量

# K值
K = 10
# 定义10个动作
ACTIONS = range(10)
VALUES = []
# 存储action-value的字典
action_value_table=defaultdict(list)

def new_bandit():
    global VALUES
    # 每一个action的真实value是N(0,1)中选取
    VALUES = [value for value in np.random.normal(0, 1, 10)]

def reset():
    global action_value_table
    action_value_table =defaultdict(list)

def return_action_reward(a):
    mu = VALUES[a]
    return np.random.normal(mu,1,10)[0]

# 根据已经做出的选择以及返回的reward计算每个acrion的值
def update_action_value(A_t,R_a):
    #将该动作的reward加入到对应的值列表中
    action_value_table[A_t].append(R_a)


# 对给定的action返回对应的value值
def action_value(a):
    reward_list = action_value_table.get(a,[])
    if len(reward_list)==0:
        #当该value不存时，对应的值是0
        Q_a = 0
    else:
        # 每个action的估计Value是其之前所有返回reward的均值
        Q_a = sum(reward_list)/float(len(reward_list))

    return Q_a

#做决策
def make_decision(delta = 0.0):
    max_as = []
    other_as = []
    max_value = -100000
    # 遍历每一个动作
    for action in ACTIONS:
        #获得当前动作的value
        value = action_value(action)
        # 如果value 大于最大值
        if value > max_value:
            max_value = value
            max_as = [action]
        #如果等于最大值，有可能很多都是0
        elif value == max_value:
            max_as.append(action)
        #如果不是最大值，加入其它列表
        else:
            other_as.append(action)

    # 生成一个随机数，来决定在哪个集合进行选择
    # 是greedy 是 delta-greedy
    rand_float = np.random.random()
    if rand_float<delta:
        selected_as = other_as
    else:
        selected_as = max_as

    # 在max_as的动作列表中选一个
    a = np.random.choice(selected_as)
    R_a = return_action_reward(a)
    #更新action value table
    update_action_value(a,R_a)

    return a,R_a

def avg_list(alist):
    return sum(alist)/float(len(alist))

#每一次的run
def run(N_LOOPS,delta,OPTIMAL_ACTION):
    #重置
    reset()
    reward_list = []
    optimal_list = [] 
    for _ in range(N_LOOPS):
        delta = 0
        action,reward = make_decision(delta=delta)
        reward_list.append(reward)
        optimal_list.append(1 if OPTIMAL_ACTION==action else 0)

    #计算每一个step完成后的平均reward以及最有决策率
    avg_reward_list = []
    optimal_rate_list = []
    for i,reward in enumerate(reward_list):
        optimals = optimal_list[:i+1]
        rewards = reward_list[:i+1]

        avg_reward_list.append(avg_list(rewards))
        optimal_rate_list.append(avg_list(optimals))

    return avg_reward_list,optimal_rate_list

if __name__ == '__main__':
    

    #设置2000组的独立实验，也就是有2000个相互独立的bandit
    #分别记录3中delta值的结果
    # 返回reward的值的大小
    _2000_runs_rewards = defaultdict(list)
    # 做出最优决策的比例
    _2000_runs_optimal_rate = defaultdict(list)
    for run_index in range(20):

        # 新的一个bandit
        new_bandit()
        # print 'ACTIONS',ACTIONS
        # print 'VALUES',VALUES
        
        # 最优决策是value最大的action
        OPTIMAL_ACTION = VALUES.index(max(VALUES))
        print run_index,"optimal action:",OPTIMAL_ACTION

        # 执行三种delta 
        delta = 0
        avg_reward_list,optimal_rate_list = run(1000,delta,OPTIMAL_ACTION)
        _2000_runs_rewards[delta].append(avg_reward_list)
        _2000_runs_optimal_rate[delta].append(optimal_rate_list)

        delta = 0.01
        avg_reward_list,optimal_rate_list = run(1000,delta,OPTIMAL_ACTION)
        _2000_runs_rewards[delta].append(avg_reward_list)
        _2000_runs_optimal_rate[delta].append(optimal_rate_list)

        delta = 0.5
        avg_reward_list,optimal_rate_list = run(1000,delta,OPTIMAL_ACTION)
        _2000_runs_rewards[delta].append(avg_reward_list)
        _2000_runs_optimal_rate[delta].append(optimal_rate_list)


    # 在完成两千个循环之后
    fig,axes = plt.subplots(2,1,figsize=(10,10))
    ax1 = axes[0]
    ax2 = axes[1]
    xs = np.array(range(1000))+1
    for delta in _2000_runs_rewards.keys():
        # plot reward
        rewards = np.array(_2000_runs_rewards[delta]).mean(axis=0)
        ax1.plot(xs,rewards,label='delta:{:}'.format(delta))

        #plot optimal
        optimals = np.array(_2000_runs_optimal_rate[delta]).mean(axis=0)
        ax2.plot(xs,optimals,label='delta:{:}'.format(delta))

    ax1.set_title("Average Rewards Trends")
    ax1.set_xlabel("Average Rewards")
    ax1.set_ylabel("time step t")

    ax2.set_title("Optimal Rate along time step t")
    ax2.set_xlabel("Optimal Rate")
    ax2.set_ylabel("time step t")

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig('10-armed.png',dpi=200)














