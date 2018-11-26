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


class Space(object):
    def __init__(self, data):
        w2id = {}
        weight = {}
        for texts in data:
            for x in texts:
                try:
                    ident = w2id[x]
                    weight[ident] = weight[ident] + 1
                except KeyError:
                    ident = len(w2id)
                    w2id[x] = ident
                    weight[ident] = 1
        self._w2id = w2id
        tot = len(weight)
        self._weight = {k: np.log(tot / v) for k, v in weight.items()}

    def doc2bow(self, tokens):
        lst = []
        w2id = self._w2id
        weight = self._weight
        for token in tokens:
            try:
                id = w2id[token]
                lst.append(id)
            except KeyError:
                continue
        ids, tf = np.unique(lst, return_counts=True)
        df = np.array([weight[x] for x in ids])
        return ids, tf, df

    def __getitem__(self, text):
        return [(i, _tf * _df) for i, _tf, _df in zip(*text)]

