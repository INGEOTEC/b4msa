# Copyright 2016 Ranyart R. Suarez (https://github.com/RanyartRodrigo) and Mario Graff (https://github.com/mgraffg)
# with collaborations of Eric S. Tellez

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
# from b4msa.textmodel import TextModel
from b4msa.utils import read_data_labels, read_data
from gensim.matutils import corpus2csc
from sklearn import preprocessing


class SVC(object):
    # def __init__(self):
    #     self.svc = LinearSVC()

    def __init__(self, model):
        self.svc = LinearSVC()
        self.model = model
        self.num_terms = -1

    def fit(self, X, y):
        X = corpus2csc(X).T
        self.num_terms = X.shape[1]
        self.le = preprocessing.LabelEncoder()
        self.le.fit(y)
        y = self.le.transform(y)
        self.svc.fit(X, y)
        return self

    def predict(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        ynew = self.svc.predict(Xnew)
        # return ynew
        return self.le.inverse_transform(ynew)
        
    def predict_text(self, text):
        y = self.predict([self.model[text]])
        print((text, y))
        return y[0]

    def fit_file(self, fname, get_tweet='text', get_klass='klass', maxitems=1e100):
        X, y = read_data_labels(fname, get_klass=get_klass, get_tweet=get_tweet, maxitems=maxitems)
        self.fit([self.model[x] for x in X], y)
        return self

    def predict_file(self, fname, get_tweet='text', maxitems=1e100):
        hy = [self.predict_text(x) for x in read_data(fname, get_tweet=get_tweet, maxitems=maxitems)]
        return hy
