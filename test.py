# 创建人: 楠
# 开发时间: 2023/7/12  10:40

from env import *
from neuralucb import *
import matplotlib.pyplot as plt
import pylab as mpl
import torch
from tqdm import trange
#期望累积收益值

N = 50
T_values = N * 24
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
    for i in range(Day_User_num):
        data = []
        Rk = np.random.uniform(100, 500)  # 均匀分布
        data.append(Rk)
        Ck = np.random.uniform(500, 1500)
        data.append(Ck)
        Fk = rd.choice(Fk_list)
        data.append(Fk)
        t_User_data.append(data)
    return t_User_data


def User_off_infor(each_Uersdata, price):
    offloadReward_data = []
    off_Usernum = 0
    for i in range((np.shape(each_Uersdata))[0]):
        if price > 1 / (each_Uersdata[i][2]):
            lk = 0
            offloadReward_data.append(lk * each_Uersdata[i][1])
        else:
            lk = (each_Uersdata[i][0] * each_Uersdata[i][1]) / (
                    each_Uersdata[i][1] + each_Uersdata[i][1] * each_Uersdata[i][2] / 10 + each_Uersdata[i][
                2] / ri)
            offloadReward_data.append(lk * each_Uersdata[i][1])
            off_Usernum += 1
    off_data = 1.35e7
    Sumoffload_data = sum(offloadReward_data)
    if Sumoffload_data <= 1.35e7:
        off_data = Sumoffload_data
    return off_data

def optimal_arm(each_Userdata):
    data = []
    for i in range(np.shape(price_list)[0]):
        reward = (User_off_infor(each_Userdata, i+1)) * (i+1)
        data.append(reward)
    return max(data)


Total_User_data = []
for i in range(N):
    for n in range(24):
        num = Total_User_num[i][n]
        Total_User_data.append(t_User_data(num))
env = IRIS()
###################################################################################################

def epsilon_greedy(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled   每只手臂被选择的次数
    rewards = [0] * k  # Cumulative rewards for each arm      每只手臂的累积奖励
    est_means = [0] * k  # Estimated mean reward for each arm  估计每只手臂的平均奖励
    regrets = []
    epsilon_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        # Calculate exploration rate epsilon for the current time step using the given theorem  利用给定的定理计算当前时间步长的勘探率
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
        optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
        regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
        regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(epsilon_Total_reward)

Total_EGreward = np.empty((experience_round, 1000))
for i in range(experience_round):
    Total_EGreward[i] = epsilon_greedy(Total_User_data, np.shape(price_list)[0], 1000)
eps_mean_reward = np.mean(Total_EGreward, axis=0)
print("eps-greedy:", eps_mean_reward[199], eps_mean_reward[399], eps_mean_reward[599], eps_mean_reward[799], eps_mean_reward[999])
###################################################################################################
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
            ucb_Total_reward.append(User_off_infor(env, price_list[arm]) * price_list[arm])
            n[arm] += 1  # Increment the count of times the chosen arm has been pulled
            rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
            est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
            optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
            regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
            regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(ucb_Total_reward)

Total_UCBreward = np.empty((experience_round, 1000))
for i in range(experience_round):
    Total_UCBreward[i] = ucb(Total_User_data, np.shape(price_list)[0], 1000)
UCB_mean_totalreward = np.mean(Total_UCBreward, axis=0)
print("UCB:", UCB_mean_totalreward[199], UCB_mean_totalreward[399], UCB_mean_totalreward[599], UCB_mean_totalreward[799], UCB_mean_totalreward[999])

######################################################################################

# Thomson_sample = TS(10)
# Total_TSreward = np.empty((experience_round, 1000))
#
# for n in range(experience_round):
#     TSreward = []
#     for i in range(1000):
#         context, reward = env.step(Total_User_data[i])
#         arm = Thomson_sample.take_action()
#         TSreward.append(User_off_infor(Total_User_data[i], price_list[arm]) * price_list[arm])
#         Thomson_sample.update(context, arm, reward[arm])
#     Sum_TSreward = np.cumsum(TSreward)
#     Total_TSreward[n] = Sum_TSreward
#
# TS_mean_totalreward = np.mean(Total_TSreward, axis=0)

######################################################################################

# linucb = LinUCB(env.dim, env.arm, beta=0.1, lamb=1)
# Total_LinUCBreward = np.empty((experience_round, 1000))
#
# for n in range(experience_round):
#     LinUCBreward = []
#     for i in range(1000):
#         context, reward = env.step(Total_User_data[i])
#         arm = linucb.take_action(context)
#         LinUCBreward.append(User_off_infor(Total_User_data[i], price_list[arm]) * price_list[arm])
#         linucb.update(context, arm, reward[arm])
#     Sum_linUCBreward = np.cumsum(LinUCBreward)
#     Total_LinUCBreward[n] = Sum_linUCBreward
#
# LinUCB_mean_totalreward = np.mean(Total_LinUCBreward, axis=0)

################################################################################

lints = offTS(env.dim, env.arm, beta=0.1, lamb=1)
Total_Lintsreward = np.empty((experience_round, 1000))

for n in range(experience_round):
    Lintsreward = []
    for i in range(1000):
        context, reward = env.step(Total_User_data[i])
        arm = lints.take_action(context)
        Lintsreward.append(User_off_infor(Total_User_data[i], price_list[arm]) * price_list[arm])
        lints.update(context, arm, reward[arm])
    Sum_lintsreward = np.cumsum(Lintsreward)
    Total_Lintsreward[n] = Sum_lintsreward

Lints_mean_totalreward = np.mean(Total_Lintsreward, axis=0)
print("off_TS:", Lints_mean_totalreward[199], Lints_mean_totalreward[399], Lints_mean_totalreward[599], Lints_mean_totalreward[799], Lints_mean_totalreward[999])

#################################################################################

# neuralucb = NeuralUCB(env.dim, env.arm, beta=0.1, lamb=1)
#
# Total_NeuralUCBreward = np.empty((experience_round, 1000))
#
# for n in range(experience_round):
#     NeuralUCBreward = []
#     for i in trange(1000):
#         context, reward = env.step(Total_User_data[i])
#         arm = neuralucb.take_action(context)
#         NeuralUCBreward.append(User_off_infor(Total_User_data[i], price_list[arm]) * price_list[arm])
#         neuralucb.update(context, arm, reward[arm])
#     Sum_NeuralUCBreward = np.cumsum(NeuralUCBreward)
#     Total_NeuralUCBreward[n] = Sum_NeuralUCBreward
#
#
# NeuralUCB_mean_totalreward = np.mean(Total_NeuralUCBreward, axis=0)
# print("NeuralUCB:", NeuralUCB_mean_totalreward[199], NeuralUCB_mean_totalreward[399], NeuralUCB_mean_totalreward[599], NeuralUCB_mean_totalreward[799], NeuralUCB_mean_totalreward[999])
#######################################################################################

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
            optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
            regret = optimal_reward - User_off_infor(env, price_list[arm]) * price_list[arm]  # Calculate regret for the chosen arm
            regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(Moss_Total_reward)



Total_Mossreward = np.empty((experience_round, 1000))
for i in range(experience_round):
    Total_Mossreward[i] = ucb(Total_User_data, np.shape(price_list)[0], 1000)
Moss_mean_totalreward = np.mean(Total_Mossreward, axis=0)
print("MOSS:", Moss_mean_totalreward[199], Moss_mean_totalreward[399], Moss_mean_totalreward[599], Moss_mean_totalreward[799], Moss_mean_totalreward[999])

#############################################################################################
def VA(Total_user, k, T):
    n = [0] * k  # Number of times each arm has been pulled   每只手臂被选择的次数
    rewards = [0] * k  # Cumulative rewards for each arm      每只手臂的累积奖励
    est_means = [0] * k  # Estimated mean reward for each arm  估计每只手臂的平均奖励
    regrets = []
    VA_Total_reward = []
    for t in range(T):
        env = Total_user[t]
        reward = User_off_infor(env, 10) / 1.35e7 # Observe the reward for the chosen arm
        VA_Total_reward.append(User_off_infor(env, k) * (k-1))
        # n[arm] += 1  # Increment the count of times the chosen arm has been pulled
        # rewards[arm] += reward  # Add the observed reward to the cumulative rewards for the chosen arm
        # est_means[arm] = rewards[arm] / n[arm]  # Update the estimated mean reward for the chosen arm
        optimal_reward = optimal_arm(env)  # Find the optimal reward among all arms
        regret = optimal_reward - User_off_infor(env, 10) * 9 # Calculate regret for the chosen arm
        regrets.append(regret)  # Add the regret to the list of regrets
    return np.cumsum(VA_Total_reward)

Total_VAreward = np.empty((experience_round, 1000))
for i in range(experience_round):
    Total_VAreward[i] = VA(Total_User_data, np.shape(price_list)[0], 1000)
VA_mean_totalreward = np.mean(Total_VAreward, axis=0)
print("VA:", VA_mean_totalreward[199], VA_mean_totalreward[399], VA_mean_totalreward[599], VA_mean_totalreward[799], VA_mean_totalreward[999])
#############################################################################################
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
size = 5
size1 = [eps_mean_reward[199], eps_mean_reward[399], eps_mean_reward[599], eps_mean_reward[799], eps_mean_reward[999]]
size2 = [UCB_mean_totalreward[199], UCB_mean_totalreward[399], UCB_mean_totalreward[599], UCB_mean_totalreward[799], UCB_mean_totalreward[999]]
# size3 = [LinUCB_mean_totalreward[199], LinUCB_mean_totalreward[399], LinUCB_mean_totalreward[599], LinUCB_mean_totalreward[799], LinUCB_mean_totalreward[999]]
size3 = [Lints_mean_totalreward[199], Lints_mean_totalreward[399], Lints_mean_totalreward[599], Lints_mean_totalreward[799], Lints_mean_totalreward[999]]
size4 = [VA_mean_totalreward[199], VA_mean_totalreward[399], VA_mean_totalreward[599], VA_mean_totalreward[799], VA_mean_totalreward[999]]
size5 = [Moss_mean_totalreward[199], Moss_mean_totalreward[399], Moss_mean_totalreward[599], Moss_mean_totalreward[799], Moss_mean_totalreward[999]]
x = np.arange(size)
x_size = ('200', '400', '600', '800', '1000')

total_width, n = 0.8, 4
width = total_width / n

x = x - (total_width - width) / 4

plt.bar(x, size1, width=width*2/3, color='white', edgecolor='k', hatch='---', label="Epsilon Greedy")
plt.bar(x + width*2/3, size2, width=width*2/3, color='white', edgecolor='k', hatch='//////', label="UCB")
plt.bar(x + 2 * width*2/3, size3, width=width*2/3, color='white', edgecolor='k', hatch='|||||', label="off_TS")
plt.bar(x + 3 * width*2/3, size4, width=width*2/3, color='black', edgecolor='k', hatch='', label="VA")
plt.bar(x + 4 * width*2/3, size5, width=width*2/3, color='white', edgecolor='k', hatch='', label="Moss")

plt.tight_layout()
plt.legend(loc='upper left', fontsize=15)
plt.xticks(x+width, x_size, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('时段', fontsize=16, labelpad=-12, x=0.99)
plt.ylabel('期望累积收益值', fontsize=16)
plt.show()