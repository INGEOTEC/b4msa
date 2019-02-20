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
from microtc.params import OPTION_NONE, get_filename, OPTION_DELETE
from microtc.weighting import Entropy
from .lang_dependency import LangDependency
import pickle
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
    >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']
    >>> textmodel = TextModel().fit(corpus)

    Represent a text into a vector

    >>> textmodel['cat']
    [(38, 0.24737436144422534), (41, 0.24737436144422534), (42, 0.4947487228884507), (73, 0.6702636255239844), (76, 0.24737436144422534), (77, 0.24737436144422534), (78, 0.24737436144422534)]
    """
    def __init__(self, docs=None, threshold=0, lang=None, negation=None, stemming=None,
                 stopwords=None, **kwargs):
        default_parameters = dict(token_list=[-2, -1, 2, 3, 4])
        self._lang_kw = dict(negation=negation, stemming=stemming, stopwords=stopwords)
        if lang:
            self.lang = LangDependency(lang)
            _ = self.default_parameters(lang=self.lang.lang)
            if _ is not None:
                default_parameters = _
                for k in self._lang_kw.keys():
                    if self._lang_kw[k] is None and k in default_parameters:
                        self._lang_kw[k] = default_parameters[k]
                    try:
                        del default_parameters[k]
                    except KeyError:
                        pass
        else:
            self.lang = False
        self._threshold = threshold
        default_parameters.update(kwargs)
        super(TextModel, self).__init__(docs, **default_parameters)

    def fit(self, X):
        """
        Train the model

        :param X: Corpus
        :type X: lst
        :rtype: instance
        """

        super(TextModel, self).fit(X)

        if self._threshold > 0:
            w = Entropy.entropy([self.tokenize(d) for d in X], X, self.model.word2id)
            self.model._w2id = {k: v for k, v in self.model._w2id.items() if w[v] > self._threshold}
        return self

    def text_transformations(self, text):
        """Language dependent transformations

        :param text: text
        :type text: str

        :rtype: str
        """

        text = super(TextModel, self).text_transformations(text)
        if self.lang:
            text = self.lang.transform(text, **self._lang_kw)

        return text

    @classmethod
    def default_parameters(self, lang=None):
        """
        Default parameters per language

        >>> from b4msa.textmodel import TextModel
        >>> TextModel.default_parameters()
        {'token_list': [-2, -1, 2, 3, 4]}
        >>> _ = TextModel.default_parameters(lang='arabic')
        >>> k = list(_.keys())
        >>> k.sort()
        >>> [(i, _[i]) for i in k]
        [('del_punc', True), ('ent_option', 'delete'), ('negation', False), ('stemming', False), ('stopwords', 'delete'), ('token_list', [-1, 2, 3, 4])]
        >>> _ = TextModel.default_parameters(lang='english')
        >>> k = list(_.keys())
        >>> k.sort()
        >>> [(i, _[i]) for i in k]
        [('del_diac', False), ('negation', False), ('num_option', 'delete'), ('stemming', False), ('stopwords', 'none'), ('token_list', [[3, 1], -2, -1, 3, 4])]
        >>> _ = TextModel.default_parameters(lang='spanish')
        >>> k = list(_.keys())
        >>> k.sort()
        >>> [(i, _[i]) for i in k]
        [('negation', False), ('stemming', False), ('stopwords', 'none'), ('token_list', [[2, 1], -1, 2, 3, 4, 5, 6])]
        """
        if lang is None:
            return dict(token_list=[-2, -1, 2, 3, 4])
        if lang == 'spanish':
            return dict(token_list=[[2, 1], -1, 2, 3, 4, 5, 6], negation=False, stemming=False, stopwords=OPTION_NONE)
        elif lang == 'english':
            return dict(token_list=[[3, 1], -2, -1, 3, 4], num_option='delete', del_diac=False, negation=False, stemming=False, stopwords=OPTION_NONE)
        elif lang == 'arabic':
            return dict(token_list=[-1, 2, 3, 4], del_punc=True, ent_option='delete', stopwords=OPTION_DELETE, negation=False, stemming=False)

    @classmethod
    def params(cls):
        """
        Parameters

        >>> from b4msa.textmodel import TextModel
        >>> TextModel.params()
        ['docs', 'threshold', 'lang', 'negation', 'stemming', 'stopwords', 'kwargs', 'docs', 'text', 'num_option', 'usr_option', 'url_option', 'emo_option', 'hashtag_option', 'ent_option', 'lc', 'del_dup', 'del_punc', 'del_diac', 'token_list', 'token_min_filter', 'token_max_filter', 'select_ent', 'select_suff', 'select_conn', 'weighting']
        """
        import inspect
        r = mTCTextModel.params()
        sig = inspect.signature(cls)
        params = sig.parameters.keys()
        return list(params) + list(r)


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
