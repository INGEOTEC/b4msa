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
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')

PUNCTUACTION = ";:,.@\\-\"'w/"
SYMBOLS = "()[]¿?¡!{}"
SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
# SKIP_WORDS = set(["…", "..", "...", "...."])

def get_word_list(text):
    text = "".join([u for u in text[1:len(text)-1] if u not in SKIP_SYMBOLS])
    return text.split('~')


def norm_chars(text, strip_diac=True, del_dup1=True):
    L = ['~']

    prev = '~'
    for u in unicodedata.normalize('NFD', text):
        if strip_diac:
            o = ord(u)
            if 0x300 <= o and o <= 0x036F:
                continue
            
        elif u in ('\n', '\r', ' ', '\t'):
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
                 lc=True,
                 del_dup1=True,
                 token_list=[1, 2, 3, 4, 5, 6, 7],
                 lang=None,
                 **kwargs
    ):
        self.strip_diac = strip_diac
        self.num_option = num_option
        self.usr_option = usr_option
        self.url_option = url_option
        self.lc = lc
        self.del_dup1 = del_dup1
        self.token_list = token_list
        if lang:
            self.lang = LangDependency(lang)
        else:
            self.lang = None
            
        self.kwargs = kwargs
        
        docs = [self.tokenize(d) for d in docs]
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(d) for d in docs]
        self.model = TfidfModel(corpus)

    def __getitem__(self, text):
        return self.model[self.dictionary.doc2bow(self.tokenize(text))]

    def language_dependent(self, text):
        if self.kwargs.get('neg', False):
            text = self.lang.negation(text)

        if self.kwargs.get('stem', False):
            text = self.lang.stemming(text)

        sw_op = self.kwargs.get('del_sw', OPTION_NONE)
        text = self.lang.filterstemming(text, sw_op)
        
        return text

    def tokenize(self, text):
        if self.lc:
            text = text.lower()

        text = norm_chars(text, self.strip_diac)

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

        if self.lang:
            text = self.language_dependent(text)
            
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
