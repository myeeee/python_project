from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[-2,6], [0,6], [0,7], [-2,5], [-3,3], [-1,0], [-2,0], [-3,1], [-1,4], [0,3], [0,1], [-1,7], [-3,5], [-4,3], [-2,0], [-3,7], [1,5], [1,2], [-2,3], [2,3], [-4,0], [-1,3], [1,1], [-2,2], [2,7], [-4,1]])
y = np.array([2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2])

# グラフ描画に向けてのデータの整形
data = np.hstack((x, y.reshape(y.shape[0], 1)))

data1 = data[np.where(data[:, 2] == 1)]
data2 = data[np.where(data[:, 2] == 2)]

# モデルの学習
model = GaussianNB()
model.fit(x, y)
print("Model fitted.")

# テストデータの用意
IndexX = np.arange(-4, 2, 0.1)
IndexY = np.arange(0, 7, 0.1)
test_data = []
for i in IndexX:
    for j in IndexY:
        test_data.append([i, j])

# 識別
test_label = model.predict(test_data)

# データ連結
test = np.hstack((test_data, test_label.reshape(test_label.shape[0], 1)))
test1 = test[np.where(test[:, 2] == 1)]
test2 = test[np.where(test[:, 2] == 2)]

# matplotlibを用いたデータの可視化（グラフ化）
plt.close("all")
#plt.scatter(data1[:, 0], data1[:, 1], c="tab:blue")
#plt.scatter(data2[:, 0], data2[:, 1], c="tab:red")
#plt.scatter(test_data[:, 0], test_data[:, 1], c="k")

plt.scatter(test1[:, 0], test1[:, 1], c="tab:blue")
plt.scatter(test2[:, 0], test2[:, 1], c="tab:red")
plt.show()

# テストデータの分類
#test_label = model.predict(test_data)
#print("Label of test data", test_data[0], ":", test_label[0])
#print("Label of test data", test_data[1], ":", test_label[1])

