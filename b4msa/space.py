# Copyright 2018 Mario Graff

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


class TFIDF(object):
    def __init__(self, docs):
        w2id = {}
        weight = {}
        self._ndocs = len(docs)
        for tokens in docs:
            for x, freq in zip(*np.unique(tokens, return_counts=True)):
                try:
                    ident = w2id[x]
                    weight[ident] = weight[ident] + 1
                except KeyError:
                    ident = len(w2id)
                    w2id[x] = ident
                    weight[ident] = 1
        self._w2id = w2id
        self._num_terms = len(w2id)
        self.wordWeight = weight

    @property
    def num_terms(self):
        """Number of terms"""

        return self._num_terms

    @property
    def wordWeight(self):
        """Word associated to each word, this could be the inverse document frequency"""
        return self._weight

    @wordWeight.setter
    def wordWeight(self, value):
        """Inverse document frequency

        :param value: dict
        """

        N = self._ndocs
        self._weight = {k: np.log2(N / v) for k, v in value.items()}

    def doc2weight(self, tokens):
        """Weight associated to each token

        :param tokes: list of tokens
        :rtype: tuple - ids, term frequency, wordWeight
        """
        lst = []
        w2id = self._w2id
        weight = self.wordWeight
        for token in tokens:
            try:
                id = w2id[token]
                lst.append(id)
            except KeyError:
                continue
        ids, tf = np.unique(lst, return_counts=True)
        tf = tf / tf.sum()
        df = np.array([weight[x] for x in ids])
        return ids, tf, df

    def __getitem__(self, text):
        """
        TF-IDF and the vectors are normalised.

        :param tuple: output of doc2weight
        :rtype: lst
        """
        
        r = [(i, _tf * _df) for i, _tf, _df in zip(*text)]
        n = np.sqrt(sum([x * x for _, x in r]))
        return [(i, x/n) for i, x in r]
