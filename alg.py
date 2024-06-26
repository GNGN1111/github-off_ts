from scipy.stats import *
import numpy as np

# 随机选取手臂
class RandomAlg:
    def __init__(self, K):
        self.K = K
    def take_action(self):
        return np.random.choice(self.K)

# UCB算法
class UCB(RandomAlg):
    def __init__(self, K, c=1):
        self.K = K
        self.c = c
        self.T = 0
        self.q = np.zeros(K)
        self.N = np.zeros(K)

    def take_action(self):
        if self.T < self.K:
            action = self.T
        else:
            action = np.argmax(self.q+self.c*np.sqrt(2*np.log(self.T)/self.N))
        self.T += 1
        return action

    def update(self, context, action, reward):
        self.q[action] = (self.q[action]*self.N[action]+reward)/(self.N[action]+1)
        self.N[action] += 1

# LinUCB算法
class LinUCB(RandomAlg):

    def __init__(self, d, K, beta=1, lamb=1):
        self.sigma_inv = lamb * np.eye(d)    # 生成对角矩阵,即生成特征向量对应的矩阵
        self.K = K                           # 手臂数目
        self.b = np.zeros((d, 1))            # 生成d行1列的矩阵
        self.beta = beta                     # lamb

    def take_action(self, context):
        theta = self.sigma_inv @ self.b
        p = np.matmul(context[:, None, :], theta) + self.beta * np.sqrt(np.matmul(np.matmul(context[:, None, :], self.sigma_inv),context[:, :, None]))
        action = np.argmax(p)
        return action

    def update(self, context, action, reward):
        self.sherman_morrison_update(context[action, :, None])
        self.b += (context[action, :, None] * (reward/1.35e7))

    def sherman_morrison_update(self, v):
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1+v.T @ self.sigma_inv @ v)


# Thompson sampling算法
class TS(RandomAlg):

    def __init__(self, K):
        self.K =K
        self.alpha = np.ones(K)
        self.beta = np.ones(K)

    def take_action(self):
        p = np.zeros(self.K)
        for k in range(self.K):
            p[k] = beta.rvs(a=self.alpha[k], b=self.beta[k])
        return np.argmax(p)

    def update(self, context, action, reward):
        if reward == 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

# 线性ts采样算法
# class LinTS(RandomAlg):
#
#     def __init__(self, d, K, beta=1, lamb=1):
#         self.sigma_inv = lamb * np.eye(d)  # 生成对角矩阵,即生成特征向量对应的矩阵
#         self.K = K
#         self.b = np.zeros((d, 1))
#         self.beta = beta
#
#     def take_action(self, context):
#         theta = multivariate_normal.rvs(mean=(self.sigma_inv @ self.b).flatten(), cov=self.beta*self.sigma_inv)
#         # 调用 rvs 方法一次会生成伪随机变量的单个值  @为矩阵乘法  .flatten()将数组将为一维
#         r_hat = np.matmul(theta[None], context[:, :, None])  # 矩阵乘法
#         return np.argmax(r_hat)
#
#     def update(self, context, action, reward):
#         self.sherman_morrison_update(context[action, :, None])
#         self.b += context[action, :, None] * reward
#
#     def sherman_morrison_update(self, v):
#         self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1 + v.T @ self.sigma_inv @ v)

class offTS(RandomAlg):

    def __init__(self, d, K, beta=1, lamb=1):
        self.sigma_inv = lamb * np.eye(d)  # 生成对角矩阵,即生成特征向量对应的矩阵
        self.K = K
        self.b = np.zeros((d, 1))
        self.beta = beta

    def take_action(self, context):
        theta = multivariate_normal.rvs(mean=(self.sigma_inv @ self.b).flatten(), cov=self.beta*self.sigma_inv)
        # 调用 rvs 方法一次会生成随机变量的对应值  @为矩阵乘法  .flatten()将数组将为一维
        # mean多维分布的均值 cov协方差矩阵
        r_hat = np.matmul(theta[None], context[:, :, None])  # 矩阵乘法
        return np.argmax(r_hat)

    def update(self, context, action, reward):
        self.sherman_morrison_update(context[action, :, None])
        self.b += context[action, :, None] * (reward / 1.35e7)

    def sherman_morrison_update(self, v):
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1 + v.T @ self.sigma_inv @ v)