import numpy as np
import matplotlib.pyplot as plt

# 学習データの生成
mu = [[0, 0], [30, 30]]
sigma = [[[20, 50], [50, 40]], [[40, -20], [-20, 40]]]
num = [2, 2]
dim = len(mu[0])
class_num = len(num)

X = np.empty([0, dim])
t = np.empty([0])

for i in range(class_num):
    X = np.append(X, np.random.multivariate_normal(mu[i], sigma[i], num[i]), axis=0)
    tSize = t.shape[0]

    for j in range(X.shape[0] - tSize):
        t = np.append(t, i)

plt.scatter(X[t == 0, 0], X[t == 0, 1], color="red", alpha=0.5)
plt.scatter(X[t == 1, 0], X[t == 1, 1], color="blue", alpha=0.5)
plt.show()


