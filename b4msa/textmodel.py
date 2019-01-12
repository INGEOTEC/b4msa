# Copyright 2016 Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from microtc.textmodel import TextModel as mTCTextModel
from microtc.params import OPTION_NONE, get_filename
from .lang_dependency import LangDependency
import pickle
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


class TextModel(mTCTextModel):
    """

    :param docs: Corpus
    :type docs: lst
    :param threshold: Threshold to remove those tokens less than 1 - entropy
    :type threshold: float
    :param lang: Language (spanish | english | italian | german)
    :type lang: str
    :param negation: Negation
    :type negation: bool
    :param stemming: Stemming
    :type stemming: bool
    :param stopwords: Stopwords (none | group | delete)
    :type stopwords: str

    Usage:

    >>> from b4msa.textmodel import TextModel
    >>> textmodel = TextModel(['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec'])

    Represent a text into a vector

    >>> textmodel['cat']
    [(38, 0.24737436144422534),
     (41, 0.24737436144422534),
     (42, 0.4947487228884507),
     (73, 0.6702636255239844),
     (76, 0.24737436144422534),
     (77, 0.24737436144422534),
     (78, 0.24737436144422534)]
    """
    def __init__(self, docs, token_list=[-2, -1, 3, 4],
                 threshold=0, lang=None, negation=False, stemming=False,
                 stopwords=OPTION_NONE, **kwargs):
        if lang:
            self.lang = LangDependency(lang)
        else:
            self.lang = False

        self._threshold = threshold
        self._lang_kw = dict(negation=negation, stemming=stemming, stopwords=stopwords)
        super(TextModel, self).__init__(docs, token_list=token_list, **kwargs)

    def fit(self, X):
        """
        Train the model

        :param X: Corpus
        :type X: lst
        :rtype: instance
        """

        super(TextModel, self).fit(X)

        if self._threshold > 0:
            w = self.entropy([self.tokenize(d) for d in X], X)
            self.model._w2id = {k: v for k, v in self.model._w2id.items() if w[v] > self._threshold}

    def entropy(self, corpus, docs):
        model = self.model
        m = model._w2id
        y = [x['klass'] for x in docs]
        klasses = np.unique(y)
        nklasses = klasses.shape[0]
        ntokens = len(m)
        weight = np.zeros((klasses.shape[0], ntokens))
        for ki, klass in enumerate(klasses):
            for _y, tokens in zip(y, corpus):
                if _y != klass:
                    continue
                for x in np.unique(tokens):
                    try:
                        weight[ki, m[x]] += 1
                    except KeyError:
                        continue
        weight = weight / weight.sum(axis=0)
        weight[~np.isfinite(weight)] = 1.0 / nklasses
        logc = np.log2(weight)
        logc[~np.isfinite(logc)] = 0
        if nklasses > 2:
            logc = logc / np.log2(nklasses)
        return (1 + (weight * logc).sum(axis=0))

    def extra_transformations(self, text):
        """Language dependent transformations

        :param text: text
        :type text: str

        :rtype: str
        """

        if self.lang:
            text = self.lang.transform(text, **self._lang_kw)

        return text


def load_model(modelfile):
    logging.info("Loading model {0}".format(modelfile))
    with open(modelfile, 'rb') as f:
        return pickle.load(f)


def get_model(basename, data, labels, args):
    modelfile = get_filename(args, os.path.join("models", os.path.basename(basename)))
    logging.info(args)

    if not os.path.exists(modelfile):
        logging.info("Creating model {0}".format(modelfile))

        if not os.path.isdir("models"):
            os.mkdir("models")

        args['docs'] = data
        model = TextModel(**args)
        with open(modelfile, 'wb') as f:
            pickle.dump(model, f)
    else:
        model = load_model(modelfile)
    return model
