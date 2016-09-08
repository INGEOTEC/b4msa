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
import re
import os
import unicodedata
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from .params import OPTION_DELETE, OPTION_GROUP, OPTION_NONE, get_filename
from .lang_dependency import LangDependency
from .utils import tweet_iterator
from collections import defaultdict
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')

PUNCTUACTION = ";:,.@\\-\"'/"
SYMBOLS = "()[]¿?¡!{}~"
SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
# SKIP_WORDS = set(["…", "..", "...", "...."])


class EmoticonClassifier:
    def __init__(self, fname=None):
        if fname is None:
            fname = os.path.join(os.path.dirname(__file__), 'resources', 'emoticons.json')

        self.emolen = defaultdict(dict)
        self.emoreg = []
        self.some = {}

        for emo in tweet_iterator(fname):
            c = emo['code'].lower()
            k = emo['klass']
            if c.isalpha():
                r = re.compile(r"\b{0}\b".format(c), re.IGNORECASE)
                self.emoreg.append((r, k))
            else:
                self.emolen[len(c)].setdefault(c, k)

            self.some[c[0]] = max(len(c), self.some.get(c[0], 0))

        maxlen = max(self.emolen.keys())
        self.emolen = [self.emolen.get(i, {}) for i in range(maxlen+1)]

    def replace(self, text, option=OPTION_GROUP):
        if option == OPTION_NONE:
            return text

        for pat, klass in self.emoreg:
            if option == OPTION_DELETE:
                klass = ''
 
            text = pat.sub(klass, text)

        T = []
        i = 0
        _text = text.lower()
        while i < len(text):
            replaced = False
            if _text[i] in self.some:
                for lcode in range(1, len(self.emolen)):
                    if i + lcode < len(_text):
                        code = _text[i:i+lcode]
                        klass = self.emolen[lcode].get(code, None)

                        if klass:
                            if option == OPTION_DELETE:
                                klass = ''

                            T.append(klass)
                            replaced = True
                            i += lcode
                            break
            
            if not replaced:
                T.append(text[i])
                i += 1

        return "".join(T)


def get_word_list(text):
    L = []
    prev = ' '
    for u in text[1:len(text)-1]:
        if u in SKIP_SYMBOLS:
            u = ' '

        if prev == ' ' and u == ' ':
            continue

        L.append(u)
        prev = u

    return ("".join(L)).split()


def norm_chars(text, strip_diac=True, del_dup1=True):
    L = ['~']

    prev = '~'
    for u in unicodedata.normalize('NFD', text):
        if strip_diac:
            o = ord(u)
            if 0x300 <= o and o <= 0x036F:
                continue
            
        if u in ('\n', '\r', ' ', '\t'):
            u = '~'

        if del_dup1 and prev == u:
            continue

        prev = u
        L.append(u)

    L.append('~')

    return "".join(L)


def expand_qgrams(text, qsize, output):
    """Expands a text into a set of q-grams"""
    n = len(text)
    for start in range(n - qsize + 1):
        output.append(text[start:start+qsize])

    return output


def expand_qgrams_word_list(wlist, qsize, output, sep='~'):
    """Expands a list of words into a list of q-grams. It uses `sep` to join words"""
    n = len(wlist)
    for start in range(n - qsize + 1):
        t = sep.join(wlist[start:start+qsize])
        output.append(t)

    return output


class TextModel:
    def __init__(self,
                 docs,
                 strip_diac=True,
                 num_option=OPTION_GROUP,
                 usr_option=OPTION_GROUP,
                 url_option=OPTION_GROUP,
                 emo_option=OPTION_GROUP,
                 lc=True,
                 del_dup1=True,
                 token_list=[-1],
                 lang=None,
                 **kwargs
    ):
        self.strip_diac = strip_diac
        self.num_option = num_option
        self.usr_option = usr_option
        self.url_option = url_option
        self.emo_option = emo_option
        self.emoclassifier = EmoticonClassifier()
        self.lc = lc
        self.del_dup1 = del_dup1
        self.token_list = token_list

        if lang:
            self.lang = LangDependency(lang)
        else:
            self.lang = None
            
        self.kwargs = {k: v for k, v in kwargs.items() if k[0] != '_'}

        docs = [self.tokenize(d) for d in docs]
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(d) for d in docs]
        self.model = TfidfModel(corpus)

    def __str__(self):
        return "[TextModel {0}]".format(dict(
            strip_diac=self.strip_diac,
            num_option=self.num_option,
            usr_option=self.usr_option,
            url_option=self.url_option,
            emo_option=self.emo_option,
            lc=self.lc,
            del_dup1=self.del_dup1,
            token_list=self.token_list,
            lang=self.lang,
            kwargs=self.kwargs
        ))

    def __getitem__(self, text):
        return self.model[self.dictionary.doc2bow(self.tokenize(text))]

    def transform_q_voc_ratio(self, text):
        tok = self.tokenize(text)
        bow = self.dictionary.doc2bow(tok)
        m = self.model[bow]
        try:
            return m, len(bow) / len(tok)
        except ZeroDivisionError:
            return m, 0

    def tokenize(self, text):
        # print("tokenizing", str(self), text)
        if text is None:
            text = ''

        if self.lc:
            text = text.lower()

        if self.num_option == OPTION_DELETE:
            text = re.sub(r"\d+\.?\d+", "", text)
        elif self.num_option == OPTION_GROUP:
            text = re.sub(r"\d+\.?\d+", "_num", text)

        if self.url_option == OPTION_DELETE:
            text = re.sub(r"https?://\S+", "", text)
        elif self.url_option == OPTION_GROUP:
            text = re.sub(r"https?://\S+", "_url", text)

        if self.usr_option == OPTION_DELETE:
            text = re.sub(r"@\S+", "", text)
        elif self.usr_option == OPTION_GROUP:
            text = re.sub(r"@\S+", "_usr", text)

        text = norm_chars(text, self.strip_diac)
        text = self.emoclassifier.replace(text, self.emo_option)

        if self.lang:
            text = self.lang.transform(text, **self.kwargs)
            
        L = []
        textlist = None

        for q in self.token_list:
            if q < 0:
                if textlist is None:
                    textlist = get_word_list(text)

                expand_qgrams_word_list(textlist, abs(q), L)
            else:
                expand_qgrams(text, q, L)
        
        return L
    

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


# if __name__ == '__main__':
#     filename = sys.argv[1]
#     from .utils
#     for kwargs in sample:
#         get_model(filename, data, labels, kwargs)
