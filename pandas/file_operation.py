from pathlib import Path
import pandas as pd

# Pathオブジェクトを生成
path_fold = Path("/---")

# Path.glob(pattern)はジェネレータを返す。結果を明示するためlist化しているが、普段は不要。
# 再帰的な検索
list_file = list(path_fold.glob("**/---*.xlsx"))

# 項目リスト
dfCol = pd.read_csv("/---.csv", encoding="shift-jis")
lisCol = list(dfCol.columns)

# dataframe初期化
all_data = pd.DataFrame()

for f in list_file:
    data = pd.read_excel(f)
    
    tmpCol = []
    for col in lisCol:
        tmpCol.append(data.columns[data.columns.str.contains(col)][0])
    
    all_data = pd.concat([all_data,  data[tmpCol]], sort=False)

all_data.to_csv('/---.csv', encoding='shift-jis')