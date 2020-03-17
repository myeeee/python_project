# 最小二乗法による線形識別
import matplotlib.pyplot as plt
import numpy as np


# 教師データの属するクラスを返す
def teaching(x, y):
    if y > x + 0.3:
        return 0
    if y > x - 0.3:
        return 1
    return 2


# 判別したクラス返す
def f(x0, x1):
    return np.argmax(np.dot(w.T, np.array([1.0, x0, x1])))


# ランダムシードを固定
np.random.seed(0)

# 訓練データ数
N = 40

# 入力次元数（ダミー入力を含めて）
M = 3

# クラス数
K = 3

# 教師データ作成
T = np.zeros([N, K])
t = np.empty(N)
xt = np.random.uniform(0, 1, N)
yt = np.random.uniform(0, 1, N)

# 外れ値
# xt[0] = 8
# yt[0] = 0.2

for i in range(N):
    t[i] = teaching(xt[i], yt[i])
    T[i, int(t[i])] = 1


# Xを作る
X = np.hstack([np.ones([N, 1]), xt.reshape([N, 1]), yt.reshape([N, 1])])

# 係数wを求める
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, T))

# グラフ表示用の判別結果
a, b = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))
vec_f = np.vectorize(f)
c = vec_f(a, b)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(xt[t == 0], yt[t == 0], color="blue", alpha=0.5)
plt.scatter(xt[t == 1], yt[t == 1], color="green", alpha=0.5)
plt.scatter(xt[t == 2], yt[t == 2], color="red", alpha=0.5)
plt.contourf(a, b, c, alpha=0.2)
plt.show()