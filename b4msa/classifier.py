# Copyright 2016 Ranyart R. Suarez (https://github.com/RanyartRodrigo) and Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.svm import LinearSVC
from b4msa.textmodel import TextModel
from b4msa.utils import tweet_iterator
from gensim.matutils import corpus2csc
from sklearn import preprocessing


class SVC(object):
    def __init__(self):
        self.svc = LinearSVC()
        self.num_terms = None

    def fit(self, fname):
        tw = [x for x in tweet_iterator(fname)]
        self.text = TextModel([x['text'] for x in tw])
        X = []
        y = []
        for l in tw:
            X.append(self.text[l['text']])
            y.append(l['klass'])
        X = corpus2csc(X).T
        self.num_terms = X.shape[1]
        self.le = preprocessing.LabelEncoder()
        self.le.fit(y)
        y = self.le.transform(y)
        self.svc.fit(X, y)
        return self

    def predict_from_file(self, fname):
        hy = [self.predict(x) for x in tweet_iterator(fname)]
        return hy

    def predict(self, newText):
        Xnew = [self.text[newText['text']]]
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        ynew = self.svc.predict(Xnew)
        return self.le.inverse_transform(ynew)[0]
