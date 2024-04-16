# 创建人: 楠
# 开发时间: 2023/11/6  15:24
import numpy
import numpy as np

from env import *
from env1 import *
from neuralucb import *
import matplotlib.pyplot as plt
import pylab as mpl
import time
from tqdm import trange

# 不同手臂数量
N = 50
T_values = N * 24
Fk_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
price_armed = [2, 4, 6, 10, 15]
# price_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ri = (1e-3) * math.log((1 + (1e-7) / 1e-9), 2)
experience_round = 10


User_num = []
for i in range(N):
    Day_User_num = np.random.poisson(lam=N, size=24)
    User_num.append(Day_User_num)
Total_User_num = np.array(User_num)

# 随机生成的移动设备计算任务信息
def t_User_data(Day_User_num):
    t_User_data = []
    each_num = Day_User_num
    for i in range(each_num):
        data = []
        Rk = np.random.uniform(100, 500)  # 均匀分布
        data.append(Rk)
        Ck = np.random.uniform(500, 1500)
        data.append(Ck)
        Fk = rd.choice(Fk_list)
        data.append(Fk)
        t_User_data.append(data)
    return t_User_data

############################################################################################
# 部分卸载个体理性的表示,即当出价是price时,MEC会收到的总卸载数据量
def User_off_infor(each_Uersdata, price):
    offloadReward_data = []
    off_Usernum = 0
    for i in range((np.shape(each_Uersdata))[0]):
        if price > 1 / (each_Uersdata[i][2]):
            lk = 0
            offloadReward_data.append(lk * each_Uersdata[i][1])
        else:
            lk = (each_Uersdata[i][0] * each_Uersdata[i][1]) / (
                    each_Uersdata[i][1] + each_Uersdata[i][1] * each_Uersdata[i][2] / 10 + each_Uersdata[i][2] / ri)
            offloadReward_data.append(lk * each_Uersdata[i][1])
            off_Usernum += 1
    off_data = 1.35e7
    Sumoffload_data = sum(offloadReward_data)
    if Sumoffload_data <= 1.35e7:
        off_data = Sumoffload_data
    return off_data


def optimal_arm(each_Userdata, k):
    price_list = np.arange(1, k+1)
    data = []
    for i in range(np.shape(price_list)[0]):
        reward = User_off_infor(each_Userdata, i+1) * (i+1)
        data.append(reward)
    return max(data)

####################################################################################################
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

Total_User_data = []
for i in range(N):
    for n in range(24):
        num = Total_User_num[i][n]
        Total_User_data.append(t_User_data(num))

env1 = IRIS1()
#####################################################################################################

def epsilon_greedy(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled   每只手臂被选择的次数
    rewards = [0] * k  # Cumulative rewards for each arm      每只手臂的累积奖励
    est_means = [0] * k  # Estimated mean reward for each arm  估计每只手臂的平均奖励
    price_list = np.arange(1, k+1)
    regrets = []
    epsilon_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        #   利用给定的定理计算当前时间步长的勘探率
        with np.errstate(divide='ignore'):
            epsilon = np.power(t, -1 / 3) * np.power(k * np.log(t), 1 / 3)  # 计算e的概率

        if np.random.rand() < epsilon:  # 如果随机值小于e则进行随机选择
            # Choose a random arm with equal probability if the exploration strategy is selected
            arm = np.random.randint(k)
        else:
            # Choose the arm with the highest estimated mean reward if the exploitation strategy is selected  不然就直接选择奖励最大的手臂
            arm = np.argmax(est_means)
        reward = User_off_infor(env, price_list[arm]) / 1.35e7 # Observe the reward for the chosen arm
        epsilon_Total_reward.append(User_off_infor(env, price_list[arm]) * price_list[arm])
        n[arm] += 1  # Increment the count of times the chosen arm has been pulled
        rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
        est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
        optimal_reward = optimal_arm(env, k)  # Find the optimal reward among all arms
        regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm] # Calculate regret for the chosen arm
        regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(epsilon_Total_reward)

time_EG_start_time = time.time()
eps_reward = np.zeros(np.shape(price_armed)[0])
for i in range(np.shape(price_armed)[0]):
    eps_rewards_ner = np.empty((experience_round, 1000))
    for n in range(experience_round):
        print(f'进入eps_reward第{i+1}轮手臂的第{n+1}实验次数')
        eps_rewards_ner[n] = epsilon_greedy(Total_User_data, price_armed[i], 1000)
    eps_mean_rewards = np.mean(eps_rewards_ner, axis=0)
    eps_reward[i] = eps_mean_rewards[999]
print(eps_reward)
plt.plot(price_armed, eps_reward, linestyle='--', marker='s',  label='Epsilon Greedy', color='black')
# time_EG_end_time = time.time()
# print('e-greedy:', eps_mean_regrets[-1])
# print("e-greedy run_time:", time_EG_end_time - time_EG_start_time)

#####################################################################################################
def ucb(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled
    rewards = [0] * k  # Cumulative rewards for each arm
    est_means = [0] * k  # Estimated mean reward for each arm
    price_list = np.arange(1, k + 1)
    regrets = []
    ucb_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        if t < k:
            # Play each arm k times to initialize the estimates and UCB values
            reward = User_off_infor(env, price_list[t]) / 1.35e7
            ucb_Total_reward.append(User_off_infor(env, price_list[t]) * price_list[t])
            n[t] += 1
            rewards[t] += reward
            est_means[t] = rewards[t] / n[t]
            regrets.append(0)
        else:
            # Choose the arm with the highest UCB value
            ucb_values = [est_means[i] + np.sqrt(2 * np.log(t) / n[i]) for i in range(k)]  # Calculate UCB values for each arm
            arm = np.argmax(ucb_values)  # Select the arm with the highest UCB value
            reward = User_off_infor(env, price_list[arm]) / 1.35e7 # Observe the reward for the chosen arm
            ucb_Total_reward.append(User_off_infor(env, price_list[arm]) * price_list[arm])
            n[arm] += 1  # Increment the count of times the chosen arm has been pulled
            rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
            est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
            optimal_reward = optimal_arm(env, k)  # Find the optimal reward among all arms
            regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
            regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(ucb_Total_reward)

ucb_reward = np.zeros(np.shape(price_armed)[0])
for i in range(np.shape(price_armed)[0]):
    ucb_rewards_ner = np.empty((experience_round, 1000))
    for n in range(experience_round):
        print(f'进入ucb第{i+1}轮手臂的第{n+1}实验次数')
        ucb_rewards_ner[n] = ucb(Total_User_data, price_armed[i], 1000)
    ucb_mean_rewards = np.mean(ucb_rewards_ner, axis=0)
    ucb_reward[i] = ucb_mean_rewards[999]
print(ucb_reward)
plt.plot(price_armed, ucb_reward, linestyle='--', marker='h', label='UCB', color='black')
# time_EG_end_time = time.time()
# print('ucb:', UCB_mean_totalregret[-1])
# print("UCB run_time:", time_UCB_end_time - time_UCB_start_time)

######################################################################################

# Thomson_sample = TS(10)
# Total_TSregret = np.empty((experience_round, 1000))
#
# for n in range(experience_round):
#     TSregret = []
#     for i in range(1000):
#         context, reward = env.step(Total_User_data[i])
#         arm = Thomson_sample.take_action()
#         chosen_arm = price_list[arm]
#         TSregret.append(optimal_arm(Total_User_data[i]) - (User_off_infor(Total_User_data[i], chosen_arm) * chosen_arm))
#         Thomson_sample.update(context, arm, reward[arm])
#     Sum_TSregret = np.cumsum(TSregret)
#     Total_TSregret[n] = Sum_TSregret
#
# TS_mean_totalregret = np.mean(Total_TSregret, axis=0)
#
# plt.plot(TS_mean_totalregret, linestyle=':', label='TS', color='black')
# print('TS:', TS_mean_totalregret[-1])
######################################################################################
time_linUCB_start_time = time.time()
linucb_reward = np.zeros(np.shape(price_armed)[0])
for m in range(np.shape(price_armed)[0]):
    linucb = LinUCB(3 * price_armed[m], price_armed[m], beta=0.1, lamb=1)
    price_list = np.arange(1, price_armed[m] + 1)
    Total_linucbreward = np.empty((experience_round, 1000))
    for n in range(experience_round):
        print(f'进入linucb第{m + 1}轮手臂的第{n + 1}实验次数')
        linucbreward = []
        for i in range(1000):
            context, reward = env1.step(Total_User_data[i], price_armed[m])
            arm = linucb.take_action(context)
            chosen_arm = price_list[arm]
            linucbreward.append(User_off_infor(Total_User_data[i], price_list[arm]) * price_list[arm])
            linucb.update(context, arm, reward[arm])
        Sum_LinUCBreward = np.cumsum(linucbreward)
        Total_linucbreward[n] = Sum_LinUCBreward
    Linucb_mean_totalreward = np.mean(Total_linucbreward, axis=0)
    linucb_reward[m] = Linucb_mean_totalreward[999]
print(linucb_reward)
# time_EG_end_time = time.time()
plt.plot(price_armed, linucb_reward, linestyle='--', marker='.', label='LinUCB')

################################################################################

time_off_ts_start_time = time.time()
offts_reward = np.zeros(np.shape(price_armed)[0])
for m in range(np.shape(price_armed)[0]):
    offts = offTS(3 * price_armed[m], price_armed[m], beta=0.1, lamb=1)
    price_list = np.arange(1, price_armed[m]+1)
    Total_offtsreward = np.empty((experience_round, 1000))
    for n in range(experience_round):
        print(f'进入off_ts第{m+1}轮手臂的第{n+1}实验次数')
        offtsreward = []
        for i in range(1000):
            context, reward = env1.step(Total_User_data[i], price_armed[m])
            arm = offts.take_action(context)
            offtsreward.append(User_off_infor(Total_User_data[i], price_list[arm]) * price_list[arm])
            offts.update(context, arm, reward[arm])
        Sum_lintsreward = np.cumsum(offtsreward)
        Total_offtsreward[n] = Sum_lintsreward
    Lints_mean_totalreward = np.mean(Total_offtsreward, axis=0)
    offts_reward[m] = Lints_mean_totalreward[999]
print(offts_reward)
# time_EG_end_time = time.time()
plt.plot(price_armed, offts_reward, linestyle='--', marker='^', label='off_TS', color='black')

# print('off_TS:', LinTS_mean_totalregret[-1])
# print("lints run_time:", time_off_ts_end_time - time_off_ts_start_time)

#################################################################################
# time_NeuralUCB_start_time = time.time()
# neuralucb_reward = np.zeros(np.shape(price_armed)[0])
# for m in range(np.shape(price_armed)[0]):
#     neuralucb = NeuralUCB(3*price_armed[m], price_armed[m], beta=0.1, lamb=1)
#     price_list = np.arange(1, price_armed[m] + 1)
#     Total_NeuralUCBreward = np.empty((experience_round, 1000))
#     for n in range(experience_round):
#         print(f'进入neuralucb第{m + 1}轮手臂的第{n + 1}实验次数')
#         NeuralUCBreward = []
#         for i in trange(1000):
#             context, reward = env1.step(Total_User_data[i], price_armed[m])
#             arm = neuralucb.take_action(context)
#             chosen_arm = price_list[arm]
#             NeuralUCBreward.append(User_off_infor(Total_User_data[i], chosen_arm) * chosen_arm)
#             neuralucb.update(context, arm, reward[arm])
#         Sum_NeuralUCBreward = np.cumsum(NeuralUCBreward)
#         Total_NeuralUCBreward[n] = Sum_NeuralUCBreward
#     NeuralUCB_mean_totalreward = np.mean(Total_NeuralUCBreward, axis=0)
#     neuralucb_reward[m] = NeuralUCB_mean_totalreward[999]
# print(neuralucb_reward)
# # time_NeuralUCB_end_time = time.time()
# plt.plot(price_armed, neuralucb_reward, linestyle='--', marker='*', label='NeuralUCB', color='black')
# # print('neuralucb:', NeuralUCB_mean_totalregret[-1])
# # print("NeuralUCB run_time:", time_NeuralUCB_end_time - time_NeuralUCB_start_time)

#################################################################################################


def Moss(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled
    rewards = [0] * k  # Cumulative rewards for each arm
    est_means = [0] * k  # Estimated mean reward for each arm
    regrets = []
    Moss_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        if t < k:
            # Play each arm k times to initialize the estimates and UCB values
            reward = User_off_infor(env, price_list[t]) / 1.35e7
            Moss_Total_reward.append(reward)
            n[t] += 1
            rewards[t] += reward
            est_means[t] = rewards[t] / n[t]
            regrets.append(0)
        else:
            # Choose the arm with the highest UCB value
            Moss_values = [est_means[i] + np.sqrt((max(T/(k*n[i]), 1)) / n[i]) for i in range(k)]  # Calculate UCB values for each arm
            arm = np.argmax(Moss_values)  # Select the arm with the highest UCB value
            reward = User_off_infor(env, price_list[arm]) / 1.35e7 # Observe the reward for the chosen arm
            Moss_Total_reward.append(User_off_infor(env, price_list[arm]) * price_list[arm])
            n[arm] += 1  # Increment the count of times the chosen arm has been pulled
            rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
            est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
            optimal_reward = optimal_arm(env, k)  # Find the optimal reward among all arms
            regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
            regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(Moss_Total_reward)

Moss_reward = np.zeros(np.shape(price_armed)[0])
for i in range(np.shape(price_armed)[0]):
    moss_rewards_ner = np.empty((experience_round, 1000))
    for n in range(experience_round):
        print(f'进入Moss第{i+1}轮手臂的第{n+1}实验次数')
        moss_rewards_ner[n] = Moss(Total_User_data, price_armed[i], 1000)
    moss_mean_rewards = np.mean(moss_rewards_ner, axis=0)
    Moss_reward[i] = moss_mean_rewards[999]
print(Moss_reward)
time_EG_end_time = time.time()
plt.plot(price_armed, Moss_reward, linestyle='--', marker='p', label='MOSS', color='black')


#############################################################################################


plt.legend(['Epsilon Greedy', 'UCB', 'off_TS', 'Moss'], fontsize=12)
plt.xlabel('价格数量', fontsize=16)
plt.ylabel('期望累积收益值', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


