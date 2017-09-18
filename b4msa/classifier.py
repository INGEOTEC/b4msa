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
import numpy as np
from b4msa.utils import read_data_labels, read_data
from gensim.matutils import corpus2csc
from sklearn import preprocessing
from sklearn import cross_validation
from b4msa.textmodel import TextModel
from multiprocessing import Pool
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


class SVC(object):
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
        if self.num_terms == 0:
            return self
        self.svc.fit(X, y)
        return self

    def decision_function(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        return self.svc.decision_function(Xnew)

    def predict(self, Xnew):
        if self.num_terms == 0:
            return self.le.inverse_transform(np.zeros(len(Xnew), dtype=np.int))
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        ynew = self.svc.predict(Xnew)
        return self.le.inverse_transform(ynew)

    def predict_text(self, text):
        y = self.predict([self.model[text]])
        return y[0]

    def fit_file(self, fname, get_tweet='text',
                 get_klass='klass', maxitems=1e100):
        X, y = read_data_labels(fname, get_klass=get_klass,
                                get_tweet=get_tweet, maxitems=maxitems)
        self.fit([self.model[x] for x in X], y)
        return self

    def predict_file(self, fname, get_tweet='text', maxitems=1e100):
        hy = [self.predict_text(x)
              for x in read_data(fname, get_tweet=get_tweet,
                                 maxitems=maxitems)]
        return hy

    @classmethod
    def predict_kfold(cls, X, y, n_folds=10, seed=0, textModel_params={},
                      kfolds=None,
                      pool=None,
                      use_tqdm=True):
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kwargs):
                return x

        le = preprocessing.LabelEncoder().fit(y)
        y = np.array(le.transform(y))
        hy = np.zeros(len(y), dtype=np.int)
        if kfolds is None:
            kfolds = cross_validation.StratifiedKFold(y,
                                                      n_folds=n_folds,
                                                      shuffle=True,
                                                      random_state=seed)
        args = [(X, y, tr, ts, textModel_params) for tr, ts in kfolds]
        if pool is not None:
            if use_tqdm:
                res = [x for x in tqdm(pool.imap_unordered(cls.train_predict_pool, args),
                                       desc='Params',
                                       total=len(args))]
            else:
                res = [x for x in pool.imap_unordered(cls.train_predict_pool, args)]
        else:
            if use_tqdm:
                args = tqdm(args)
            res = [cls.train_predict_pool(x) for x in args]
        for ts, _hy in res:
            hy[ts] = _hy
        return le.inverse_transform(hy)

    @classmethod
    def train_predict_pool(cls, args):
        X, y, tr, ts, textModel_params = args
        t = TextModel([X[x] for x in tr], **textModel_params)
        m = cls(t).fit([t[X[x]] for x in tr], [y[x] for x in tr])
        return ts, np.array(m.predict([t[X[x]] for x in ts]))

    @classmethod
    def predict_kfold_params(cls, fname, n_folds=10, score=None, numprocs=None, seed=0, param_kwargs={}):
        from b4msa.params import ParameterSelection, Wrapper
        X, y = read_data_labels(fname)
        if numprocs is not None:
            pool = Pool(numprocs)
        else:
            pool = None
            numprocs = 1

        if n_folds % numprocs == 0:
            f = Wrapper(X, y, score, n_folds, cls, pool=pool, seed=seed)
            pool = None
        else:
            f = Wrapper(X, y, score, n_folds, cls, seed=seed)

        return ParameterSelection().search(f.f, pool=pool, **param_kwargs)

    @classmethod
    def fit_from_file(cls, fname, textModel_params={}):
        X, y = read_data_labels(fname)
        model = TextModel(X, **textModel_params)
        svc = cls(model)
        return svc.fit([model[x] for x in X], y)
