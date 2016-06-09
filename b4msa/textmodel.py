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
from .params import OPTION_DELETE, OPTION_GROUP, get_filename
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


def norm_chars(text, strip_diac=True):
    L = ['~']

    for u in unicodedata.normalize('NFD', text):
        if strip_diac:
            o = ord(u)
            if 0x300 <= o and o <= 0x036F:
                continue
        elif u in ('\n', '\r', ' ', '\t'):
            u = '~'

        L.append(u)
    L.append('~')

    return "".join(L)


def expand_qgrams(text, qsize, output):
    """Expands a text into a set of q-grams"""
    n = len(text)
    for start in range(n - qsize + 1):
        output.append(text[start:start+qsize])

    return output


class TextModel:
    def __init__(self,
                 docs,
                 strip_diac=True,
                 usr_option=OPTION_GROUP,
                 url_option=OPTION_GROUP,
                 lc=True,
                 token_list=[1, 2, 3, 4, 5, 6, 7]
    ):
        self.strip_diac = strip_diac
        self.usr_option = usr_option
        self.url_option = url_option
        self.lc = lc
        self.token_list = token_list
        docs = [self.tokenize(d) for d in docs]
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(d) for d in docs]
        self.model = TfidfModel(corpus)

    def __getitem__(self, text):
        return self.model[self.dictionary.doc2bow(self.tokenize(text))]

    def language_dependent(self, text):
        return text

    def tokenize(self, text):
        if self.lc:
            text = text.lower()

        text = norm_chars(text, self.strip_diac)

        if self.url_option == OPTION_DELETE:
            text = re.sub("https?://\S+", "", text)
        elif self.url_option == OPTION_GROUP:
            text = re.sub("https?://\S+", "_url", text)

        if self.usr_option == OPTION_DELETE:
            text = re.sub("@\S+", "", text)
        elif self.usr_option == OPTION_GROUP:
            text = re.sub("@\S+", "_usr", text)

        text = self.language_dependent(text)
            
        L = []
        for q in self.token_list:
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
