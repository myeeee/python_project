import numpy as np


class Perceptron:

    def __init__(self, eta=0.01, n_iter=50, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):

        gen = np.random.RandomState(self.seed)
        self.w_ = gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            error = 0

            for x, target in zip(X, y):

                update = self.eta * (target - self.predict(x))

                # 重みの更新
                self.w_[1:] += update * x
                self.w_[0] += update

                error += int(update != 0.0)

            self.errors_.append(error)

        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

