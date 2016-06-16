# Copyright 2016 Eric S. Tellez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from time import time
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from b4msa.classifier import SVC
# from b4msa.utils import read_data_labels, tweet_iterator
from b4msa.utils import tweet_iterator
from b4msa.textmodel import TextModel
from multiprocessing import Pool, cpu_count
# import json
import numpy as np
from b4msa.params import ParameterSelection
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')

LABELMAP = {
    1: 'POS',
    2: 'NEU',
    3: 'NEG',
    4: 'NONE',
    '_negativo': 'NEG',
    '_neutro': 'NEU',
    '_positivo': 'POS',
    '_desconocido': 'NONE',
    'N': 'NEG',
    'N+': 'NEG',
    'P': 'POS',
    'P+': 'POS',
    'POS': 'POS',
    'NEG': 'NEG',
    'NEU': 'NEU',
    'NONE': 'NONE',
}

class Wrapper(object):
    def __init__(self, text1, y1, text2, y2, cls, seed=0, pool=None):
        self.text1 = text1
        self.text2 = text2
        self.y1 = y1 = [LABELMAP[y] for y in y1]
        self.y2 = y2 = [LABELMAP[y] for y in y2]
        le = preprocessing.LabelEncoder().fit(y1 + y2)
        self.y1 = np.array(le.transform(y1))
        self.y2 = np.array(le.transform(y2))
        self.cls = cls
        self.pool = pool
        np.random.seed(seed)

    def f(self, conf_code):
        conf, code = conf_code
        st = time()
        model = TextModel(self.text1, **conf)
        C = self.cls(model)
        C.fit([model[text] for text in self.text1], self.y1)
        conf['_fit_time'] = time() - st
        st = time()
        hy = C.predict([model[text] for text in self.text2])
        conf['_predict_time'] = time() - st

        conf['_macro_f1'] = f1_score(self.y2, hy, average='macro')
        conf['_weighted_f1'] = f1_score(self.y2, hy, average='weighted')
        conf['_accuracy'] = accuracy_score(self.y2, hy)
        conf['_score'] = conf['_macro_f1']
       
        return conf


def map_label(tweet):
    tweet['klass'] = LABELMAP[tweet['klass']]
    return tweet
    
def main(trainname, testname, bsize=16, qsize=3, hill_climbing=True, numprocs=None, seed=0, num_klasses=2):
    assert num_klasses in (2, 4), "The number of classes num_klasses should be 2 or 4"
    
    if num_klasses == 4:
        accepted_klasses = ['POS', 'NEG', 'NEU', 'NONE']
    else:
        accepted_klasses = ['POS', 'NEG']
        
    A = [map_label(tweet) for tweet in tweet_iterator(trainname)]
    B = [map_label(tweet) for tweet in tweet_iterator(testname)]
    x1, y1 = [], []
    for tweet in A:
        if tweet['klass'] in accepted_klasses:
            x1.append(tweet['text'])
            y1.append(tweet['klass'])

    x2, y2 = [], []
    for tweet in B:
        if tweet['klass'] in accepted_klasses:
            x2.append(tweet['text'])
            y2.append(tweet['klass'])

    logging.info("train: {0}, test: {1}", len(x1), len(x2))
    f = Wrapper(x1, y1, x2, y2, SVC, seed=seed)

    numprocs = cpu_count() if numprocs == 0 else numprocs
    return ParameterSelection().search(f.f,
                                       bsize=bsize,
                                       qsize=qsize,
                                       hill_climbing=hill_climbing,
                                       pool=Pool(numprocs))


if __name__ == '__main__':
    from b4msa.command_line import CommandLine

    class PA(CommandLine):
        def __init__(self):
            self.parser = argparse.ArgumentParser(description='b4msa')
            # self.training_set()
            pa = self.parser.add_argument
            pa('training_set', nargs=2, default=None, help="training-set test-set")
            pa('-c', '--classes', dest='num_klasses', default=2, type=int,
               help="The number of classes in the task. The valid values are 2 or 4 (for [NEG, POS] and [NEG, POS, NEU, NONE] respectively)")

            self.param_set()
            self.param_search()

        def get_output(self):
            if self.data.output is None:
                return self.data.training_set[0] + ".output"
            
            return self.data.output
    
        def main(self):
            import json
            self.data = self.parser.parse_args()
            best_list = main(self.data.training_set[0],
                             self.data.training_set[1],
                             bsize=self.data.samplesize,
                             qsize=self.data.qsize,
                             hill_climbing=self.data.hill_climbing,
                             numprocs=self.data.numprocs,
                             num_klasses=self.data.num_klasses)

            with open(self.get_output(), 'w') as f:
                f.write(json.dumps(best_list, indent=2, sort_keys=True))

    pa = PA()
    pa.main()

