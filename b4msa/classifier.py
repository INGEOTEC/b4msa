# Copyright 2016 Mario Graff (https://github.com/mgraffg)

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
from b4msa.textmodel import TextModel, tweet_iterator


class Classifier(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class SVC(object):
    def __init__(self):
        self.svc = LinearSVC()

    def fit(self, fname):
        tw = [x for x in tweet_iterator(fname)]
        self.text = TextModel([x['text'] for x in tw])
        X = []
        y = []
        for l in tw:
            # print(l)
            X.append(self.text[l['text']])
            y.append(l['klass'])
        X = corpus2csc(self.X).T
        y = np.argmax(label_binarize(self.y, classes=['NEG','NEU','POS']),axis=1)
        svc.fit(X, y)
        return self

    def predict(self, newText):
        Xnew = []
        tw_new = [x for x in tweet_iterator(newText)]
        for l in tw_new:
            Xnew.append(self.text[l['text']])
        Xnew = corpus2csc(Xnew).T
        ynew = svc.predict(Xnew)
        return ynew
