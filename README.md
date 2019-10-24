# PMI
自己相互情報量をpandasのデータフレームから計算

## 概要
pandasのデータフレームから単語の入った列を指定して、全列のPMIを計算、PMI降順に単語を指定数引き出してpmi列に格納

## 使い方

```python
from pmi_calc import PmiExtractor
import pandas as pd

pe = PmiExtractor(kw_num=3, word_column="内容")
df = pd.read_csv("test.tsv", delimiter="\t")
df = pe.make_pmi(df, 3)
# df["pmi"]にpmi降順に抜き出された単語が格納
```

## 引数
### PmiExtractor
kw_num: 指定してキーワード数、単語を格納

word_column: データフレームの文章が格納されている列名

dic_location: 利用する辞書のパス

### make_pmi
df: データフレーム

n_gram: 指定した数字のn-gramで計算
