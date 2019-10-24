import MeCab
import math
import numpy as np

from collections import OrderedDict


class PmiExtractor(object):
    def __init__(
        self,
        kw_num,
        word_column,
        dic_location="/usr/local/lib/mecab/dic/mecab-ipadic-neologd",
    ):
        self.kw_num = kw_num
        self.word_column = word_column
        self.pmi_matrix = OrderedDict()
        self.index2word = OrderedDict()
        self.word2index = OrderedDict()
        self.m_w = MeCab.Tagger("-d %s -Owakati" % dic_location)

    # mecabで分かち書き
    def _owakati(self, text):
        return self.m_w.parse(text).split(" ")

    # indexにするための辞書作成
    def _makedict(self, list_text):
        i = 0
        for sentence in list_text:
            for word in sentence:
                if word not in self.word2index:
                    self.word2index[word] = i
                    i += 1

        for k, v in self.word2index.items():
            self.index2word[v] = k

    # 文字をindex化
    def _to_index(self, text):
        index_list = []
        for word in text:
            index_list.append(self.word2index[word])
        return index_list

    # 共起行列、PMI行列を作成
    def _pmi(self, df, n_gram):
        # 2次元行列の最大値を取得
        max_index = 0
        for index in df["word_index"]:
            for i in index:
                if max_index < i:
                    max_index = i

        # 共起行列
        co_occurrence_matrix = OrderedDict()
        co_occurrence_matrix = [
            [0 for i in range(max_index + 1)] for j in range(max_index + 1)
        ]
        for _, row in df.iterrows():
            for i, word in enumerate(row["word_index"]):
                for j, word2 in enumerate(
                    row["word_index"][
                        i + 1
                        if i + 1 <= len(row["word_index"])
                        else len(row["word_index"]) : i + 1 + n_gram
                        if i + 1 + n_gram <= len(row["word_index"])
                        else len(row["word_index"])
                    ]
                ):  # 文末でループしないようなn_gram
                    co_occurrence_matrix[word][word2] += 1
                    co_occurrence_matrix[word2][word] += 1

        # pmi行列を作成
        self.pmi_matrix = [
            [0 for i in range(max_index + 1)] for j in range(max_index + 1)
        ]
        for i in range(max_index + 1):
            for j in range(max_index + 1):
                if i is not j:
                    if co_occurrence_matrix[i][j] != 0:
                        tmp = math.log(
                            (co_occurrence_matrix[i][j] * max_index)
                            / (
                                sum(co_occurrence_matrix[i])
                                * sum(co_occurrence_matrix[j])
                            ),
                            2,
                        )
                        # 0以上の時のみ代入
                        self.pmi_matrix[i][j] = tmp if 0 < tmp else 0
                        self.pmi_matrix[j][i] = tmp if 0 < tmp else 0

    # キーワードを抽出
    def _index_to_pmi(self, df):
        df["pmi"] = np.nan
        for iloc_num, (_, row) in enumerate(df.iterrows()):
            pmi_list = []
            kw = []
            for num, i in enumerate(row["word_index"]):
                for j in row["word_index"][num + 1 :]:
                    pmi_list.append([i, j, self.pmi_matrix[i][j]])
            pmi_list = sorted(pmi_list, key=lambda x: x[2], reverse=True)
            i = 0
            while len(kw) < self.kw_num:  # キーワードの数が指定の数を満たすまでループ
                try:
                    len(pmi_list) - 1 > i
                except IndexError:
                    break
                if pmi_list[i][0] not in kw:
                    kw.append(pmi_list[i][0])
                if pmi_list[i][1] not in kw:
                    kw.append(pmi_list[i][1])
                i += 1
            kw_list = [self.index2word[word] for word in kw]
            df["pmi"].iloc[iloc_num] = kw_list[: self.kw_num]
        return df

    def make_pmi(self, df, n_gram):
        df["word_index"] = df[self.word_column].apply(self._owakati)
        self._makedict(df["word_index"])
        df["word_index"] = df["word_index"].apply(self._to_index)
        self._pmi(df, n_gram)
        df = self._index_to_pmi(df)
        return df
