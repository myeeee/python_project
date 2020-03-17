import numpy as np
import matplotlib.pyplot as plt
import sys
import CreateData

# |x|<ZERO_EPS のとき, x=0が成立していると見なす.
ZERO_EPS = 1.0e-2
# SMO法の反復回数上限
MAX_ITER = 1000

np.random.seed(92)
'''
N = 4

train_x = np.zeros((N, 2))
train_t = np.zeros(N, dtype=int)

train_x[0:int(N/2), :] = np.random.normal([0.5, 0], [0.2, 0.4], (int(N/2), 2))
train_x[int(N/2):N, :] = np.random.normal([-0.5, 0], [0.2, 0.4], (int(N-N/2), 2))
train_t[0:int(N/2)] = 1
train_t[int(N/2):N] = -1


X = train_x
t = train_t
'''
dim = 2
X = np.empty([0, dim])
t = np.empty([0])

# 学習データの作成
#############################################################################
#x, label = CreateData.create([50, 50], [[20, 50], [50, 40]], 5, -1)
x, label = CreateData.create([-0.5, 0], [[0.1, 0.2], [0.2, 0.5]], 5, 1)
X = np.append(X, x, axis=0)
t = np.append(t, label)

x, label = CreateData.create([0.5, 0], [[0.1, 0.2], [0.2, 0.5]], 5, -1)
X = np.append(X, x, axis=0)
t = np.append(t, label)
#############################################################################


# データ数
Data_Num = X.shape[0]
# ラグランジュ未定乗数初期化
alpha = np.random.uniform(0, 1, t.shape)

def check_kkt(alpha, theta, i):
    wo = opt_w(alpha)
    yi = t[i] * (wo.T.dot(X[i]) + theta)

    return yi >= 1 and alpha[i] * yi < ZERO_EPS


def opt_w(alpha):
    w = np.zeros(X[0].shape)
    for i in range(alpha.shape[0]):
        if alpha[i] < ZERO_EPS:
            continue
        w += alpha[i] * t[i] * X[i]
    return w


def choose_sec(alpha, i):
    wo = opt_w(alpha)
    yi = wo.T.dot(X[i])
    max_val = 0
    index = i
    for j in range(Data_Num):
        if alpha[j] < ZERO_EPS:
            continue
        yj = wo.T.dot(X[j])
        val = abs(yj - yi)
        if val > max_val:
            max_val = val
            index = j
    return index


def threshold(alpha):
    cnt = 0
    avg = 0
    wo = opt_w(alpha)

    for i in range(Data_Num):
        if alpha[i] < ZERO_EPS:
            continue
        avg += t[i] - wo.T.dot(X[i])
        cnt += 1
    return avg / cnt


def update_alpha(alpha, p, q):
    if p == q:
        return False
    wo = opt_w(alpha)
    fp = wo.T.dot(X[p])
    fq = wo.T.dot(X[q])
    delta_p = (1 - t[p] * t[q] + t[p] * (fq - fp)) / (X[p].T.dot(X[p]) - 2 * X[p].T.dot(X[q]) + X[q].T.dot(X[q]))
    next_ap = alpha[p] + delta_p
    c = t[p] * alpha[p] + t[q] * alpha[q]
    if t[p] == t[q]:
        next_ap = min(max(0.0, next_ap), c / t[p])
    else:
        next_ap = max(max(0.0, c / t[p]), next_ap)
    if abs(next_ap - alpha[p]) < ZERO_EPS:
        return False
    alpha[p] = next_ap
    alpha[q] = (c - t[p] * alpha[p]) / t[q]

    return True


# バイアスの初期化
theta = threshold(alpha)

for itr in range(MAX_ITER):
    changed = False
    for n1 in range(Data_Num):
        if check_kkt(alpha, theta, n1):
            continue
        n2 = choose_sec(alpha, n1)
        changed = update_alpha(alpha, n1, n2) or changed
        theta = threshold(alpha)
    if not changed:
        break


a, b = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
#a, b = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
vec_f = np.vectorize(lambda ai, bi: opt_w(alpha).T.dot(np.array([ai, bi])) + theta)
c = vec_f(a, b)
c = c <= 0


plt.scatter(X[t == -1, 0], X[t == -1, 1], color="red", alpha=0.5)
plt.scatter(X[t == 1, 0], X[t == 1, 1], color="blue", alpha=0.5)
plt.contourf(a, b, c, alpha=0.2)
plt.show()

