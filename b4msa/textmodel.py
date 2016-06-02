import re
import os
import sys
import shutil
import json
import unicodedata
import gzip
import random
import logging
import numpy as np
from gensim import corpora, models
from gensim.models.tfidfmodel import TfidfModel
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tweet_iterator(filename):
    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename)
    else:
        f = open(filename, encoding='ascii')

    while True:
        line = f.readline()
        if type(line) is bytes:
            line = str(line, encoding='ascii')
        if len(line) == 0:
            break

        line = line.strip()
        if len(line) == 0:
            continue

        # print(line)
        t = None
        try:
            t = get_tweet(line)
            yield t
        except (json.decoder.JSONDecodeError, ValueError):
            print("WARNING! we found and error while parsing file:", filename)
            print("most of these errors occur due to concurrent writes")

    f.close()


def get_tweet(line):
    return json.loads(line)


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


OPTION_NONE = 'none'
OPTION_GROUP = 'group'
OPTION_DELETE = 'delete'

class TextModel:
    def __init__(self,
                 docs,
                 strip_diac=True,
                 usr_option=OPTION_GROUP,
                 url_option=OPTION_GROUP,
                 lc=True,
                 token_list=[1, 2, 3, 4, 5, 6, 7],
                 language_dependent=[]
    ):
        self.strip_diac = strip_diac
        self.usr_option = usr_option
        self.url_option = url_option
        self.lc = lc
        self.token_list = token_list
        self.language_dependent = language_dependent
        docs = [self.tokenize(d) for d in docs]
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(d) for d in docs]
        self.model = TfidfModel(corpus)

    def __getitem__(self, text):
        return self.model[self.dictionary.doc2bow(self.tokenize(text))]
        
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

        for ldep in self.language_dependent:
            text = ldep(text)
            
        L = []
        for q in self.token_list:
            expand_qgrams(text, q, L)

        return L


def get_filename(basename, kwargs):
    L = [basename]
    for k, v in sorted(kwargs.items()):
        L.append("{0}={1}".format(k, v).replace(" ", ""))

    return "-".join(L)
    

def load_model(modelfile):
    logging.info("Loading model {0}".format(modelfile))
    with open(modelfile, 'rb') as f:
        return pickle.load(f)


def get_model(basename, data, labels, args):
    modelfile = get_filename(os.path.join("models", os.path.basename(basename)), args)
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


if __name__ == '__main__':
    filename = sys.argv[1]
    params = dict(
        strip_diac=[False, True],
        usr_option=[OPTION_DELETE, OPTION_GROUP, OPTION_NONE],
        url_option=[OPTION_DELETE, OPTION_GROUP, OPTION_NONE],
        lc=[False, True],
        token_list=[1, 2, 3, 4, 5, 6, 7],
        language_dependent=[]
    )

    data, labels = [], []
    for i, tweet in enumerate(tweet_iterator(filename)):
        data.append(tweet['text'])
        labels.append(tweet['klass'])

        if random.random() < 0.005:
            logging.info("#{1}, text: {0}".format(data[-1], i+1))

        if i == 10000:
            break

    sample = []
    # for i in range(128):
    np.random.seed(0)  # just to produce *locally* reproducible performances
    for i in range(10):
        kwargs = {}
        for k, v in sorted(params.items()):
            if len(v) == 0:
                continue
    
            if k == 'token_list':
                # kwargs[k] = sorted(np.random.sample(v, 3))
                x = list(v)
                np.random.shuffle(x)
                kwargs[k] = sorted(x[:3])
            else:
                kwargs[k] = np.random.choice(v)

        sample.append(kwargs)

    for kwargs in sample:
        get_model(filename, data, labels, kwargs)
