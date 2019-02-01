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
from b4msa.utils import read_data_labels, read_data, tweet_iterator
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from b4msa.textmodel import TextModel
from multiprocessing import Pool
from scipy.sparse import csr_matrix
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


class SVC(object):
    """Classifier

    :param model: TextModel
    :type model: class

    Usage:

    >>> from b4msa.textmodel import TextModel
    >>> from b4msa.classifier import SVC
    >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']
    >>> textmodel = TextModel(corpus)
    >>> svc = SVC(textmodel)
    >>> _ = svc.fit([textmodel[x] for x in corpus], [1, 0, 0])
    >>> svc.predict_text('hola')
    0
    """
    def __init__(self, model, **kwargs):
        self.svc = LinearSVC(**kwargs)
        self.model = model

    @property
    def num_terms(self):
        """Dimension which is the number of terms of the corpus

        :rtype: int
        """

        try:
            return self._num_terms
        except AttributeError:
            self._num_terms = None
        return None

    def tonp(self, X):
        """Sparse representation to sparce matrix

        :param X: Sparse representation of matrix
        :type X: list
        :rtype: csr_matrix
        """

        data = []
        row = []
        col = []
        for r, x in enumerate(X):
            cc = [_[0] for _ in x if np.isfinite(_[1]) and (self.num_terms is None or _[0] < self.num_terms)]
            col += cc
            data += [_[1] for _ in x if np.isfinite(_[1]) and (self.num_terms is None or _[0] < self.num_terms)]
            _ = [r] * len(cc)
            row += _
        if self.num_terms is None:
            _ = csr_matrix((data, (row, col)))
            self._num_terms = _.shape[1]
            return _
        return csr_matrix((data, (row, col)), shape=(len(X), self.num_terms))

    def fit(self, X, y):
        """Train the classifier

        :param X: inputs - independent variables
        :type X: lst
        :param y: output - dependent variable

        :rtype: instance
        """

        X = self.tonp(X)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(y)
        y = self.le.transform(y)
        if self.num_terms == 0:
            return self
        self.svc.fit(X, y)
        return self

    def decision_function(self, Xnew):
        Xnew = self.tonp(Xnew)
        return self.svc.decision_function(Xnew)

    def predict(self, Xnew):
        if self.num_terms == 0:
            return self.le.inverse_transform(np.zeros(len(Xnew), dtype=np.int))
        Xnew = self.tonp(Xnew)
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
                      kfolds=None, pool=None, use_tqdm=True):
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kwargs):
                return x

        le = preprocessing.LabelEncoder().fit(y)
        y = np.array(le.transform(y))
        hy = np.zeros(len(y), dtype=np.int)
        if kfolds is None:
            kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=seed).split(X, y)
        args = [(X, y, tr, ts, textModel_params) for tr, ts in kfolds]
        if pool is not None:
            if use_tqdm:
                res = [x for x in tqdm(pool.imap_unordered(cls.train_predict_pool, args),
                                       desc='Params', total=len(args))]
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
        params = TextModel.params()
        textModel_params = {k: v for k, v in textModel_params.items() if k in params}
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
        D = [x for x in tweet_iterator(fname)]
        # X, y = read_data_labels(fname)
        y = [x['klass'] for x in D]
        model = TextModel(D, **textModel_params)
        svc = cls(model)
        return svc.fit([model[x] for x in D], y)
