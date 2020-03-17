import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
data = pd.read_csv('/---.csv', encoding='shift-jis')

# 1列目削除
data = data.iloc[:, 1:]

# datetimeインデックスに変換
data['---'] = pd.to_datetime(data['---'])

# ---・日付をインデックスに指定
data = data.set_index(['---'])

# x軸目盛
dtrange = pd.date_range('---', '---', freq='M')
# dataframeリスト
dftmp = []
keylist = []

for key, grp in data.groupby('---'):
    dftmp.append(grp.resample('M').count()['---'].reindex(dtrange).fillna(0))
    keylist.append(key)

ind = 0
dftmp[ind].plot.bar(width=1, label=keylist[ind], color='C{}'.format(ind))
dfsum = dftmp[ind]

for ind in range(1, len(dftmp)):
    dftmp[ind].plot.bar(width=1, label=keylist[ind], color='C{}'.format(ind), bottom=dfsum)
    dfsum = dfsum + dftmp[ind]

plt.legend()
plt.show()
