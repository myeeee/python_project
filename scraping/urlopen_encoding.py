import sys
from urllib.request import urlopen

f = urlopen('http://sample.scraping-book.com/dp')

# HTTPヘッダーからエンコーディングを取得する（明示されていない場合はutf-8とする）。

encoding = f.info().get_content_charset(failobj="utf-8")
print('encoding:', encoding, file=sys.stderr)   # エンコーディングを標準エラー出力に出力する。

text = f.read().decode(encoding)
print(text)
