import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
import sys

def createData(mu, sigma, num,):
    # 2次元正規乱数の生成
    values = multivariate_normal(mu, sigma, num)
    # 列ベクトル(要素1)作成
    ones = np.ones(values.shape[0])
    ones = ones[:, np.newaxis]
    # 連結
    data = np.hstack([ones, values])
    return data

mu1 = [0, 0]
sigma1 = [[20, 50], [50, 40]]
num1 = 50
mu2 = [30, 30]
sigma2 = [[40, -20], [-20, 40]]
num2 = 50
mu3 = [50, -20]
sigma3 = [[30, -15], [-15, 30]]
num3 = 50

# 2次元正規乱数を作成
data1 = createData(mu1, sigma1, num1)
data2 = createData(mu2, sigma2, num2)
data3 = createData(mu3, sigma3, num3)

label1 = np.empty((0, 3))
print(data1.shape)
for i in range(data1.shape[0]):
    label1 = np.append(label1, np.array([[1, 0, 0]]), axis=0)

label2 = np.empty((0, 3))
for i in range(data2.shape[0]):
    label2 = np.append(label2, np.array([[0, 1, 0]]), axis=0)

label3 = np.empty((0, 3))
for i in range(data3.shape[0]):
    label3 = np.append(label3, np.array([[0, 0, 1]]), axis=0)

data = np.concatenate([data1, data2, data3])
label = np.concatenate([label1, label2, label3])

w = np.dot(data.T, data)
w = np.linalg.inv(w)
w = np.dot(w, data.T)
w = np.dot(w, label)

x = np.linspace(-20, 60, 50)
y = np.linspace(-20, 60, 50)
for i in range(len(x)):
    for j in range(len(y)):
        f1 = y[j] * w[0, 2] + x[i] * w[0, 1] + w[0, 0]
        f2 = y[j] * w[1, 2] + x[i] * w[1, 1] + w[1, 0]
        f3 = y[j] * w[2, 2] + x[i] * w[2, 1] + w[2, 0]
        maximum = max([f1, f2, f3])
        if maximum == f1:
            color = "k"
        elif maximum == f2:
            color = "c"
        elif maximum == f3:
            color = "y"
        plt.scatter(x[i], y[j], c = color)

# w0 + xw1 + yw2 = 0
#y = (x * w[0, 1] + w[0, 0]) / (-w[0, 2])
#y = (w[1, 1] - w[0, 1]) * x + (w[1, 0] + w[0, 0]) / (w[0, 2] - w[1, 2])
#plt.plot(x, y)
#y = (x * w[1, 1] + w[1, 0]) / (-w[1, 2])
#y = (w[2, 1] - w[1, 1]) * x + (w[2, 0] + w[1, 0]) / (w[1, 2] - w[2, 2])
#lt.plot(x, y)
#y = (x * w[2, 1] + w[2, 0]) / (-w[2, 2])
#y = (w[0, 1] - w[2, 1]) * x + (w[0, 0] + w[2, 0]) / (w[2, 2] - w[0, 2])
#plt.plot(x, y)

plt.scatter(data1[:, 1], data1[:, 2])
plt.scatter(data2[:, 1], data2[:, 2])
plt.scatter(data3[:, 1], data3[:, 2])

plt.show()

