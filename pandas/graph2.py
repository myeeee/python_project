import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
data = pd.read_csv("/---.csv", encoding='shift-jis', index_col=0)

# datetimeインデックスに変換
data['---'] = pd.to_datetime(data['---'])

# ---・日付をインデックスに指定
data = data.set_index(['---'])

#print(data.groupby('---').resample('M').count())

for key, grp in data.groupby('---'):
    grp.resample('M').count()['---'].reindex(pd.date_range('---', '---', freq='M')).fillna(0).plot(label=key)

plt.grid(which='both')
plt.xticks(range(int(plt.xlim()[0]), int(plt.xlim()[1]) + 1), pd.date_range('---', '---', freq='M').strftime('%Y-%b'), rotation=90)
plt.legend()
plt.show()