import numpy as np


# 学習データの作成
def create(mu, sigma, num, label):
    dim = len(mu)

    x = np.empty([0, dim])
    t = np.empty([0])

    x = np.append(x, np.random.multivariate_normal(mu, sigma, num), axis=0)
    for i in range(x.shape[0]):
        t = np.append(t, label)

    return x, t
