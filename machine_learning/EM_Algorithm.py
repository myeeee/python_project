import numpy as np
import matplotlib.pyplot as plt
from sympy import zeta

import CreateData
from numpy import linalg as la
import sys

#np.random.seed(50)

class_num = 3
dim = 2


X = np.empty([0, dim])
# 学習データの作成
#############################################################################
x, label = CreateData.create([2.5, 0.1], [[0.1, -0.5], [-0.5, 0.5]], 100, 1)
X = np.append(X, x, axis=0)

x, label = CreateData.create([-0.5, 0.1], [[0.1, -0.5], [-0.5, 0.5]], 100, 1)
X = np.append(X, x, axis=0)

x, label = CreateData.create([0.5, 0], [[0.1, 0.5], [0.5, 0.5]], 100, -1)
X = np.append(X, x, axis=0)
#############################################################################

#X = np.array([[-1,-1],[-1,0],[0,1],[1,1],[1,2],[2,3],[11,5],[6,2],[1,7],[0,4],[10,7],[-1,10],[-1,2],[10,10],[11,0],[0,4],[20,20],[19,19],[19,18],[18,19]])

def make_gauss(muk, sigmak, i):
    gap = X[i] - muk
    inv = la.inv(sigmak)
    det = la.det(sigmak)
    gaussk = np.exp(-0.5 * gap.T.dot(inv).dot(gap)) / ((2*np.pi)**(1/dim) * np.sqrt(abs(det)))
    return gaussk


def calc_likelihood():
    log_like = 0
    for i in range(data_num):
        for k in range(class_num):
            if z[i] != k:
                continue
            log_like += np.log(pi[k] * make_gauss(mu[k], sigma[k], i))
    return log_like

data_num = X.shape[0]

mu = np.random.uniform(X.max(), X.min(), (class_num, dim))
sigma = np.array([[[5, 1], [1, 5]], [[5, 1], [1, 5]], [[5, 1], [1, 5]]])
#sigma = np.random.uniform(0.5*np.var(X), 1.5*np.var(X), (class_num, dim, dim))
pi = np.ones(class_num) / X.shape[0]
'''
#mu = [np.array([0, 0]), np.array([1, 0])]
mu = np.array([[0, 0], [1, 0]])
#sigma = [np.eye(2), np.eye(2)]
sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
pi = [0.5, 0.5]
'''
z = np.random.randint(0, class_num, data_num)
pre_like = calc_likelihood()

cnt=0
while True:

    # E-step
    gamma = np.empty([X.shape[0], class_num])
    for i in range(X.shape[0]):
        denomin = 0
        for k in range(class_num):
            denomin += pi[k] * make_gauss(mu[k], sigma[k], i)
        for k in range(class_num):
            gamma[i][k] = pi[k] * make_gauss(mu[k], sigma[k], i) / denomin


    # M-Step
    N = np.zeros(class_num)

    for i in range(data_num):
        for k in range(class_num):
            N[k] += gamma[i][k]

    mu = np.zeros(mu.shape)

    for i in range(data_num):
        for k in range(class_num):
            mu[k] += gamma[i][k] * X[i]

    for k in range(class_num):
        mu[k] = mu[k] / N[k]

    sigma = np.zeros(sigma.shape)

    for k in range(class_num):
        for i in range(data_num):
            mt_gap = np.reshape(X[i]-mu[k], (dim, 1))
            sigma[k] += gamma[i][k] * np.dot(mt_gap, mt_gap.T)
    for k in range(class_num):
        sigma[k] = sigma[k] / N[k]

    pi = N / data_num

    # クラス分類
    for i in range(data_num):
        z[i] = np.argmax(gamma[i])

    # 収束判定
    like = calc_likelihood()
    print(pre_like-like)
    if(abs(pre_like-like) < 0.01):
    #if np.allclose(pre_like, like, ):
        break
    pre_like = like
    cnt += 1


plt.scatter(X[z == 0, 0], X[z == 0, 1], color="red", alpha=0.5)
plt.scatter(X[z == 1, 0], X[z == 1, 1], color="blue", alpha=0.5)
plt.scatter(X[z == 2, 0], X[z == 2, 1], color="green", alpha=0.5)
plt.show()

