import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Perceptron
import Common

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#print(df.tail())

y = df.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[:100, [0, 2]].values

ppn = Perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

Common.plot_decision_region(X, y, ppn)

plt.show()



