import numpy as np

from env import *
from neuralucb import *
import matplotlib.pyplot as plt
import pylab as mpl
import time
from tqdm import trange
#期望累积遗憾值

N = 50
# T_values = N * 24
Fk_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
price_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ri = (1e-3) * math.log((1 + (1e-7) / 1e-9), 2)
experience_round = 10

User_num = []
for i in range(N):
    Day_User_num = np.random.poisson(lam=50, size=24)
    User_num.append(Day_User_num)
Total_User_num = np.array(User_num)


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
    off_data = 0
    Sumoffload_data = sum(offloadReward_data)
    if Sumoffload_data <= 1.35e7:
        off_data = Sumoffload_data
    if Sumoffload_data > 1.35e7:
        off_data = 1.35e7
    return off_data

# MEC收到的卸载人数与请求人数的比率，我们将这个比率作为MAB算法接收到的reward,reward属于[0,1]
# def User_off_infor_rate(each_Uersdata, price):
#     offloadReward_data = []
#     init_reward_data = []
#     off_Usernum = 0
#     for i in range((np.shape(each_Uersdata))[0]):
#         if price > 1 / (each_Uersdata[i][2]):
#             lk = 0
#             offloadReward_data.append(lk * each_Uersdata[i][1])
#         else:
#             lk = (each_Uersdata[i][0] * each_Uersdata[i][1]) / (
#                     each_Uersdata[i][1] + each_Uersdata[i][1] * each_Uersdata[i][2] / 10 + each_Uersdata[i][2] / ri)
#             offloadReward_data.append(lk * each_Uersdata[i][1])
#             init_reward_data.append(each_Uersdata[i][0] * each_Uersdata[i][1])
#             off_Usernum += 1
#     return   off_Usernum / (np.shape(each_Uersdata))[0]

def optimal_arm(each_Userdata):
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

env = IRIS()
#####################################################################################################

def epsilon_greedy(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled   每只手臂被选择的次数
    rewards = [0] * k  # Cumulative rewards for each arm      每只手臂的累积奖励
    est_means = [0] * k  # Estimated mean reward for each arm  估计每只手臂的平均奖励
    regrets = []
    epsilon_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        # print(env)
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
        epsilon_Total_reward.append(reward)
        n[arm] += 1  # Increment the count of times the chosen arm has been pulled
        rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
        est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
        optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
        regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm] # Calculate regret for the chosen arm
        regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(regrets)

time_EG_start_time = time.time()
eps_regrets_ner = np.empty((experience_round, 1000))
for i in range(experience_round):
    eps_regrets_ner[i] = epsilon_greedy(Total_User_data, np.shape(price_list)[0], 1000)
eps_mean_regrets = np.mean(eps_regrets_ner, axis=0)
time_EG_end_time = time.time()
# print(eps_mean_regrets[199], eps_mean_regrets[399], eps_mean_regrets[599], eps_mean_regrets[799], eps_mean_regrets[999])
plt.plot(eps_mean_regrets, linestyle='--', marker='', label='EG')
print('e-greedy:', eps_mean_regrets[-1])
print("e-greedy run_time:", time_EG_end_time - time_EG_start_time)

#####################################################################################################
def ucb(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled
    rewards = [0] * k  # Cumulative rewards for each arm
    est_means = [0] * k  # Estimated mean reward for each arm
    regrets = []
    ucb_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        if t < k:
            # Play each arm k times to initialize the estimates and UCB values
            reward = User_off_infor(env, price_list[t]) / 1.35e7
            ucb_Total_reward.append(reward)
            n[t] += 1
            rewards[t] += reward
            est_means[t] = rewards[t] / n[t]
            regrets.append(0)
        else:
            # Choose the arm with the highest UCB value
            ucb_values = [est_means[i] + np.sqrt(2 * np.log(t) / n[i]) for i in range(k)]  # Calculate UCB values for each arm
            arm = np.argmax(ucb_values)  # Select the arm with the highest UCB value
            reward = User_off_infor(env, price_list[arm]) / 1.35e7 # Observe the reward for the chosen arm
            ucb_Total_reward.append(reward)
            n[arm] += 1  # Increment the count of times the chosen arm has been pulled
            rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
            est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
            optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
            regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
            regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(regrets)

time_UCB_start_time = time.time()
Total_UCBregret = np.empty((experience_round, 1000))
for i in range(experience_round):
    Total_UCBregret[i] = ucb(Total_User_data, np.shape(price_list)[0], 1000)
UCB_mean_totalregret = np.mean(Total_UCBregret, axis=0)
time_UCB_end_time = time.time()
# print(UCB_mean_totalregret[199], UCB_mean_totalregret[399], UCB_mean_totalregret[599], UCB_mean_totalregret[799], UCB_mean_totalregret[999])
plt.plot(UCB_mean_totalregret, linestyle='--', label='UCB', color='black')
print('ucb:', UCB_mean_totalregret[-1])
print("UCB run_time:", time_UCB_end_time - time_UCB_start_time)

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
# linucb = LinUCB(env.dim, env.arm, beta=0.1, lamb=1)
# Total_LinUCBregret = np.empty((experience_round, 1000))
# time_linUCB_start_time = time.time()
# for n in range(experience_round):
#     LinUCBregret = []
#     for i in range(1000):
#         context, reward = env.step(Total_User_data[i])
#         arm = linucb.take_action(context)
#         chosen_arm = price_list[arm]
#         LinUCBregret.append(optimal_arm(Total_User_data[i])-(User_off_infor(Total_User_data[i], chosen_arm) * chosen_arm))
#         linucb.update(context, arm, reward[arm])
#     Sum_LinUCBregret = np.cumsum(LinUCBregret)
#     Total_LinUCBregret[n] = Sum_LinUCBregret
#
# LinUCB_mean_totalregret = np.mean(Total_LinUCBregret, axis=0)
# time_linUCB_end_time = time.time()
# print(LinUCB_mean_totalregret[199], LinUCB_mean_totalregret[399], LinUCB_mean_totalregret[599], LinUCB_mean_totalregret[799], LinUCB_mean_totalregret[999])
# plt.plot(LinUCB_mean_totalregret, linestyle='--', label='LinUCB')
# print('linucb:', LinUCB_mean_totalregret[-1])
# # print("linUCB run_time:", time_linUCB_end_time - time_linUCB_start_time)

################################################################################

lints = offTS(env.dim, env.arm, beta=0.1, lamb=1)
Total_LinTSregret = np.empty((experience_round, 1000))

time_lints_start_time = time.time()
for n in range(experience_round):
    LinTSregret = []
    for i in range(1000):
        context, reward = env.step(Total_User_data[i])
        # print(f"第{i+1}个context:{context}")
        arm = lints.take_action(context)
        chosen_arm = price_list[arm]
        LinTSregret.append(optimal_arm(Total_User_data[i])-(User_off_infor(Total_User_data[i], chosen_arm) * chosen_arm))
        lints.update(context, arm, reward[arm])
    Sum_LinTSregret = np.cumsum(LinTSregret)
    Total_LinTSregret[n] = Sum_LinTSregret

LinTS_mean_totalregret = np.mean(Total_LinTSregret, axis=0)
time_lints_end_time = time.time()
# print(LinTS_mean_totalregret[199], LinTS_mean_totalregret[399], LinTS_mean_totalregret[599], LinTS_mean_totalregret[799], LinTS_mean_totalregret[999])
plt.plot(LinTS_mean_totalregret, linestyle='--', label='off_TS', color='green')
print('off_TS:', LinTS_mean_totalregret[-1])
print("lints run_time:", time_lints_end_time - time_lints_start_time)

#################################################################################

# neuralucb = NeuralUCB(env.dim, env.arm, beta=0.1, lamb=1)
# Total_NeuralUCBregret = np.empty((experience_round, 1000))
# time_NeuralUCB_start_time = time.time()
# for n in range(experience_round):
#     NeuralUCBregret = []
#     for i in trange(1000):
#         context, reward = env.step(Total_User_data[i])
#         arm = neuralucb.take_action(context)
#         chosen_arm = price_list[arm]
#         NeuralUCBregret.append(optimal_arm(Total_User_data[i])-(User_off_infor(Total_User_data[i], chosen_arm) * chosen_arm))
#         neuralucb.update(context, arm, reward[arm])
#     Sum_NeuralUCBregret = np.cumsum(NeuralUCBregret)
#     Total_NeuralUCBregret[n] = Sum_NeuralUCBregret
#
# NeuralUCB_mean_totalregret = np.mean(Total_NeuralUCBregret, axis=0)
# time_NeuralUCB_end_time = time.time()
# print(NeuralUCB_mean_totalregret[199], NeuralUCB_mean_totalregret[399], NeuralUCB_mean_totalregret[599], NeuralUCB_mean_totalregret[799], NeuralUCB_mean_totalregret[999])
# plt.plot(NeuralUCB_mean_totalregret, linestyle='--', label='NeuralUCB')
# print('neuralucb:', NeuralUCB_mean_totalregret[-1])
# # print("NeuralUCB run_time:", time_NeuralUCB_end_time - time_NeuralUCB_start_time)
#######################################################################################

def Moss(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled
    rewards = [0] * k  # Cumulative rewards for each arm
    est_means = [0] * k  # Estimated mean reward for each arm
    regrets = []
    ucb_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        if t < k:
            # Play each arm k times to initialize the estimates and UCB values
            reward = User_off_infor(env, price_list[t]) / 1.35e7
            ucb_Total_reward.append(reward)
            n[t] += 1
            rewards[t] += reward
            est_means[t] = rewards[t] / n[t]
            regrets.append(0)
        else:
            # Choose the arm with the highest UCB value
            ucb_values = [est_means[i] + np.sqrt((max(T/(k*n[i]), 1)) / n[i]) for i in range(k)]  # Calculate UCB values for each arm
            arm = np.argmax(ucb_values)  # Select the arm with the highest UCB value
            reward = User_off_infor(env, price_list[arm]) / 1.35e7 # Observe the reward for the chosen arm
            ucb_Total_reward.append(reward)
            n[arm] += 1  # Increment the count of times the chosen arm has been pulled
            rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
            est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
            optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
            regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
            regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(regrets)

time_Moss_start_time = time.time()
Total_Mossregret = np.empty((experience_round, 1000))
for i in range(experience_round):
    Total_Mossregret[i] = Moss(Total_User_data, np.shape(price_list)[0], 1000)
MOSS_mean_totalregret = np.mean(Total_Mossregret, axis=0)
time_Moss_end_time = time.time()
# print(MOSS_mean_totalregret[199], MOSS_mean_totalregret[399], MOSS_mean_totalregret[599], MOSS_mean_totalregret[799], MOSS_mean_totalregret[999])
plt.plot(MOSS_mean_totalregret, linestyle='--', label='Moss')
print('Moss:', MOSS_mean_totalregret[-1])
print("Moss run_time:", time_Moss_end_time - time_Moss_start_time)


#############################################################################################

def VA(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled   每只手臂被选择的次数
    rewards = [0] * k  # Cumulative rewards for each arm      每只手臂的累积奖励
    est_means = [0] * k  # Estimated mean reward for each arm  估计每只手臂的平均奖励
    regrets = []
    epsilon_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        reward = User_off_infor(env, 10) / 1.35e7 # Observe the reward for the chosen arm
        epsilon_Total_reward.append(reward)
        n[arm] += 1  # Increment the count of times the chosen arm has been pulled
        rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
        est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
        optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
        regret = optimal_reward - User_off_infor(env, 10) * 9 # Calculate regret for the chosen arm
        regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(regrets)

time_VA_start_time = time.time()
VA_regrets_ner = np.empty((experience_round, 1000))
for i in range(experience_round):
    VA_regrets_ner[i] = VA(Total_User_data, np.shape(price_list)[0], 1000)
VA_mean_regrets = np.mean(VA_regrets_ner, axis=0)
time_VA_end_time = time.time()
# print(VA_mean_regrets[199], VA_mean_regrets[399], VA_mean_regrets[599], VA_mean_regrets[799], VA_mean_regrets[999])
plt.plot(VA_mean_regrets, linestyle='--', marker='', label='VA')
print('VA:', VA_mean_regrets[-1])
print("VA耗时:", time_VA_end_time - time_VA_start_time)



###########################################################################################
plt.legend(['e-greedy', 'UCB', 'off_TS', 'MOSS', 'VA'], fontsize=15)
plt.xlabel('时段', fontsize=16)
plt.ylabel('期望累积遗憾值', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()