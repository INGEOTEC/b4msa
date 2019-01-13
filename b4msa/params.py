# author: Eric S. Tellez <eric.tellez@infotec.mx>
# under the same terms than the multilingual benchmark

import numpy as np
import logging
from time import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')

OPTION_NONE = 'none'
OPTION_GROUP = 'group'
OPTION_DELETE = 'delete'


BASIC_OPTIONS = [OPTION_DELETE, OPTION_GROUP, OPTION_NONE]

_BASE_PARAMS = dict(
    del_diac=[False, True],
    num_option=BASIC_OPTIONS,
    usr_option=BASIC_OPTIONS,
    url_option=BASIC_OPTIONS,
    emo_option=BASIC_OPTIONS,
    lc=[False, True],
    del_dup=[False, True],
    # knowledge=[False, True],
    token_list=[-2, -1, 1, 2, 3, 4, 5, 6, 7],
)

_BASE_PARAMS_LANG = dict(
    del_diac=[False, True],
    num_option=BASIC_OPTIONS,
    usr_option=BASIC_OPTIONS,
    url_option=BASIC_OPTIONS,
    emo_option=BASIC_OPTIONS,
    lc=[False, True],
    del_dup=[False, True],
    # knowledge=[False, True],
    token_list=[-2, -1, 1, 2, 3, 4, 5, 6, 7],
    negation=[False, True],
    stemming=[False, True],
    stopwords=BASIC_OPTIONS,
)

BASE_PARAMS = sorted(_BASE_PARAMS.items())
BASE_PARAMS_LANG = sorted(_BASE_PARAMS_LANG.items())


class ParameterSelection:
    def __init__(self):
        self.lang = None

    def sample_param_space(self, n, q=3):
        for i in range(n):
            kwargs = dict(lang=self.lang)
            for k, v in self.base_params:
                if len(v) == 0:
                    continue

                if k == 'token_list':
                    x = list(v)
                    np.random.shuffle(x)
                    # qs = np.random.randint(q, len(x))
                    qs = int(round(min(len(x), max(1, np.random.normal(q)))))
                    kwargs[k] = sorted(x[:qs])
                else:
                    kwargs[k] = v[np.random.randint(len(v))]

            yield kwargs

    def expand_neighbors(self, s):
        for k, v in sorted(s.items()):
            if k[0] == '_' or k == 'lang':
                # by convention, metadata starts with underscore
                continue
            
            if v in (True, False):
                x = s.copy()
                x[k] = not v
                yield x
            elif v in BASIC_OPTIONS:
                for _v in BASIC_OPTIONS:
                    if _v != v:
                        x = s.copy()
                        x[k] = _v
                        yield x
            elif k == 'token_list':
                for i in range(len(v)):
                    x = s.copy()
                    l = x[k] = x[k].copy()
                    l.pop(i)
                    yield x

                for _v in self._base_params[k]:
                    if _v not in v:
                        x = s.copy()
                        l = x[k] = x[k].copy()
                        l.append(_v)
                        l.sort()
                        yield x

    def search(self, fun_score, bsize=32, qsize=3,
               hill_climbing=True, lang=None, pool=None):

        self.lang = lang
        if lang:
            self.base_params = BASE_PARAMS_LANG
            self._base_params = _BASE_PARAMS_LANG
        else:
            self.base_params = BASE_PARAMS
            self._base_params = _BASE_PARAMS

        tabu = set()  # memory for tabu search

        # initial approximation, montecarlo based process
        def get_best(cand, desc="searching for params"):
            if pool is None:
                # X = list(map(fun_score, cand))
                X = [fun_score(x) for x in tqdm(cand, desc=desc, total=len(cand))]
            else:
                # X = list(pool.map(fun_score, cand))
                X = [x for x in tqdm(pool.imap_unordered(fun_score, cand), desc=desc, total=len(cand))]

            # a list of tuples (score, conf)
            X.sort(key=lambda x: x['_score'], reverse=True)
            return X

        L = []
        for conf in self.sample_param_space(bsize, q=qsize):
            code = get_filename(conf)
            if code in tabu:
                continue

            tabu.add(code)
            L.append((conf, code))

        best_list = get_best(L)
        if hill_climbing:
            # second approximation, a hill climbing process
            i = 0
            while True:
                i += 1
                bscore = best_list[0]['_score']
                L = []

                for conf in self.expand_neighbors(best_list[0]):
                    code = get_filename(conf)
                    if code in tabu:
                        continue

                    tabu.add(code)
                    L.append((conf, code))

                best_list.extend(get_best(L, desc="hill climbing iteration {0}".format(i)))
                best_list.sort(key=lambda x: x['_score'], reverse=True)
                if bscore == best_list[0]['_score']:
                    break

        return best_list


class Wrapper(object):
    def __init__(self, X, y, score, n_folds, cls, seed=0, pool=None):
        self.n_folds = n_folds
        self.score = score
        self.X = X
        self.le = le = preprocessing.LabelEncoder().fit(y)
        self.y = np.array(le.transform(y))
        self.cls = cls
        self.pool = pool
        np.random.seed(seed)
        self.kfolds = [x for x in StratifiedKFold(n_splits=n_folds, shuffle=True,
                                                  random_state=seed).split(np.zeros(self.y.shape[0]),
                                                                           self.y)]

    def f(self, conf_code):
        conf, code = conf_code
        st = time()
        hy = self.cls.predict_kfold(self.X, self.y, self.n_folds,
                                    textModel_params=conf,
                                    kfolds=self.kfolds,
                                    pool=self.pool,
                                    use_tqdm=False)
        self.compute_score(conf, hy)
        conf['_time'] = (time() - st) / self.n_folds
        return conf

    def compute_score(self, conf, hy):
        RS = recall_score(self.y, hy, average=None)
        conf['_all_f1'] = M = {str(self.le.inverse_transform([klass])[0]): f1 for klass, f1 in enumerate(f1_score(self.y, hy, average=None))}
        conf['_all_recall'] = {str(self.le.inverse_transform([klass])[0]): f1 for klass, f1 in enumerate(RS)}
        conf['_all_precision'] = N = {str(self.le.inverse_transform([klass])[0]): f1 for klass, f1 in enumerate(precision_score(self.y, hy, average=None))}
        conf['_macrorecall'] = np.mean(RS)
        if len(self.le.classes_) == 2:
            conf['_macrof1'] = np.mean(np.array([v for v in conf['_all_f1'].values()]))
            conf['_weightedf1'] = conf['_microf1'] = f1_score(self.y, hy, average='binary')
        else:
            conf['_macrof1'] = f1_score(self.y, hy, average='macro')
            conf['_microf1'] = f1_score(self.y, hy, average='micro')
            conf['_weightedf1'] = f1_score(self.y, hy, average='weighted')
        conf['_accuracy'] = accuracy_score(self.y, hy)
        if self.score.startswith('avgf1:'):
            _, k1, k2 = self.score.split(':')
            conf['_' + self.score] = (M[k1] + M[k2]) / 2
        elif self.score.startswith('avgf1f0:'):
            _, k1, k2 = self.score.split(':')
            pos = (M[k1] + N[k1]) / 2.
            neg = (M[k2] + N[k2]) / 2.
            conf['_' + self.score] = (pos + neg) / 2.
        conf['_score'] = conf['_' + self.score]


def get_filename(kwargs, basename=None):
    L = []
    if basename:
        L.append(basename)
        
    for k, v in sorted(kwargs.items()):
        L.append("{0}={1}".format(k, v).replace(" ", ""))

    return "-".join(L)
