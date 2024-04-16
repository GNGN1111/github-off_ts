import numpy as np
import pandas as pd
import random as rd
import math


class IRIS():
    def __init__(self):
        self.arm = 10   # 手臂数目
        self.dim = 30  # 上下文特征向量维度
        # self.data = pd.read_csv('Iris.csv')  # 读取文件数据

    # 50天内按照泊松分布到达Ap点处的用户数目

    def step(self, User_data):
        ri = (1e-3) * math.log((1 + (1e-7) / 1e-9), 2)
        price_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        def User_off_infor(each_Uersdata, price):
            offloadReward_data = []
            init_reward_data = []
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
                    init_reward_data.append(each_Uersdata[i][0] * each_Uersdata[i][1])
                    off_Usernum += 1
            off_data = 0
            Sumoffload_data = sum(offloadReward_data)
            if Sumoffload_data <= 1.35e7:
                off_data = Sumoffload_data
            off = []
            off.append(off_data)
            off.append(off_Usernum/(np.shape(each_Uersdata))[0])
            off.append(price)
            return off

        price = rd.choice(price_list)
        X_n = []
        for i in range(10):
            front = np.zeros((3 * i))
            back = np.zeros((3 * (9 - i)))
            x = User_off_infor(User_data, price)
            new_d = np.concatenate((front, x, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)

        reward = np.zeros(self.arm)
        reward[price-1] = (User_off_infor(User_data, price))[0] * price
        return X_n, reward
