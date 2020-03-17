import pandas as pd
import matplotlib.pyplot as plt
import seaborn

data = pd.read_csv('/---.csv', encoding='shift-jis')

# 列削除
data_tmp = data.drop(data.columns[1:4], axis=1)

# 1列目をインデックスに指定
data_tmp = data_tmp.set_index(data_tmp.columns[0])

# インデックス別合計
data_tmp = data_tmp.sum(level=0)

# 転置
data_tmp = data_tmp.T
print(data_tmp)

data_tmp.plot(grid=True)
plt.xticks(list(range(0, len(data_tmp.index))), data_tmp.index, rotation=90)
plt.show()
