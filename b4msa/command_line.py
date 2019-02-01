# Copyright 2016 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import b4msa
from b4msa.classifier import SVC
from b4msa.utils import read_data, tweet_iterator, read_data_labels
from b4msa.textmodel import TextModel
# from b4msa.params import OPTION_DELETE
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import json
import gzip
import pickle

# from b4msa.params import ParameterSelection


def clean_params(kw):
    params = TextModel.params()
    return {k: v for k, v in kw.items() if k in params}


def load_json(filename):
    if filename.endswith('.gz'):
        func = gzip.open
    else:
        func = open
    with func(filename, 'rb') as fpt:
        try:
            d = fpt.read()
            return json.loads(str(d, encoding='utf-8'))
        except TypeError:
            return json.loads(d)


class CommandLine(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.training_set()
        self.predict_kfold()
        self.param_set()
        self.param_search()
        self.langdep()
        self.version()

    def version(self):
        pa = self.parser.add_argument
        pa('--version',
           action='version', version='B4MSA %s' % b4msa.__version__)

    def predict_kfold(self):
        pa = self.parser.add_argument
        pa('-k', '--kfolds', dest='n_folds',
           help='Predict the training set using stratified k-fold',
           type=int)

    def training_set(self):
        cdn = 'File containing the training set on csv.'
        pa = self.parser.add_argument
        pa('training_set',
           # nargs=1,
           default=None,
           help=cdn)
        pa('--verbose', dest='verbose', type=int,
           help='Logging level default: INFO+1',
           default=logging.INFO+1)

    def langdep(self):
        pa = self.parser.add_argument
        pa('-l', '--lang', dest='lang', type=str, default=None,
           help="Language (spanish|english|italian) to be use in the language-dependent features. Default is None, i.e., use only language-independent features")

    def param_search(self):
        pa = self.parser.add_argument
        pa('-s', '--sample', dest='samplesize', type=int,
           default=8,
           help="The sample size of the parameter")
        # pa('-q', '--qsize', dest='qsize', type=int, default=3,
        #   help="The size of the initial population of tokenizers")
        pa('-H', '--hillclimbing', dest='hill_climbing', default=False,
           action='store_true',
           help="Determines if hillclimbing search is also perfomed to improve the selection of paramters")
        pa('-n', '--numprocs', dest='numprocs', type=int, default=1,
           help="Number of processes to compute the best setup")
        pa('-S', '--score', dest='score', type=str, default='macrorecall',
           help="The name of the score to be optimized (macrorecall|macrof1|weightedf1|accuracy|avgf1:klass1:klass2); it defaults to macrof1")

    def param_set(self):
        pa = self.parser.add_argument
        pa('-o', '--output-file', dest='output',
           help='File name to store the output')
        pa('--seed', default=0, type=int)

    def get_output(self):
        if self.data.output is None:
            return self.data.training_set + ".output"
        return self.data.output

    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        logger = logging.getLogger('b4msa')
        logger.setLevel(self.data.verbose)
        if self.data.numprocs == 1:
            numprocs = None
        elif self.data.numprocs == 0:
            numprocs = cpu_count()
        else:
            numprocs = self.data.numprocs

        n_folds = self.data.n_folds
        n_folds = n_folds if n_folds is not None else 5
        assert self.data.score.split(":")[0] in ('macrorecall', 'macrof1', 'microf1', 'weightedf1', 'accuracy', 'avgf1', 'avgf1f0'), "Unknown score {0}".format(self.data.score)

        best_list = SVC.predict_kfold_params(
            self.data.training_set,
            n_folds=n_folds,
            score=self.data.score,
            numprocs=numprocs,
            seed=self.data.seed,
            param_kwargs=dict(
                bsize=self.data.samplesize,
                hill_climbing=self.data.hill_climbing,
                # qsize=self.data.qsize,
                lang=self.data.lang
            )
        )
        output = self.get_output()
        if output.endswith('.gz'):
            with gzip.open(output, 'wb') as fpt:
                cdn = json.dumps(best_list, indent=2, sort_keys=True)
                fpt.write(bytes(cdn, encoding='utf-8'))
        else:
            with open(output, 'w') as fpt:
                fpt.write(json.dumps(best_list, indent=2, sort_keys=True))


class CommandLineTrain(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.param_set()
        self.training_set()
        self.param_train()
        self.version()

    def param_train(self):
        pa = self.parser.add_argument
        pa('-m', '--model-params', dest='params_fname', type=str,
           required=False, help="TextModel params")
        pa('--kw', dest='kwargs', default=None, type=str,
           help='Parameters in json that overwrite b4msa default parameters')

    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        logger = logging.getLogger('b4msa')
        logger.setLevel(self.data.verbose)
        params_fname = self.data.params_fname
        if params_fname is not None:
            best = load_json(params_fname)
            if isinstance(best, list):
                best = best[0]
        else:
            best = dict()
        best = clean_params(best)
        kw = json.loads(self.data.kwargs) if self.data.kwargs is not None else dict()
        best.update(kw)
        svc = SVC.fit_from_file(self.data.training_set, best)
        with open(self.get_output(), 'wb') as fpt:
            pickle.dump(svc, fpt)


class CommandLineTest(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.param_set()
        self.training_set()
        self.param_test()
        self.version()

    def param_test(self):
        pa = self.parser.add_argument
        pa('-m', '--model', dest='model', type=str,
           required=True,
           help="SVM Model file name")
        pa('--decision-function', dest='decision_function', default=False,
           action='store_true',
           help='Outputs the decision functions instead of the class')

    def training_set(self):
        cdn = 'File containing the test set'
        pa = self.parser.add_argument
        pa('test_set',
           default=None,
           help=cdn)
        pa('--verbose', dest='verbose', type=int,
           help='Logging level default: INFO+1',
           default=logging.INFO+1)

    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        logger = logging.getLogger('b4msa')
        logger.setLevel(self.data.verbose)
        with open(self.data.model, 'rb') as fpt:
            svc = pickle.load(fpt)
        X = [svc.model[x] for x in read_data(self.data.test_set)]
        output = self.get_output()
        if output.endswith('.gz'):
            gzip_flag = True
            output = gzip.open(output, 'wb')
        else:
            gzip_flag = False
            output = open(output, 'w')
        with output as fpt:
            if not self.data.decision_function:
                hy = svc.predict(X)
                for tweet, klass in zip(tweet_iterator(self.data.test_set), hy):
                    tweet['klass'] = str(klass)
                    cdn = json.dumps(tweet)+"\n"
                    cdn = bytes(cdn, encoding='utf-8') if gzip_flag else cdn
                    fpt.write(cdn)
            else:
                hy = svc.decision_function(X)
                for tweet, klass in zip(tweet_iterator(self.data.test_set), hy):
                    try:
                        o = klass.tolist()
                    except AttributeError:
                        o = klass
                    tweet['decision_function'] = o
                    cdn = json.dumps(tweet)+"\n"
                    cdn = bytes(cdn, encoding='utf-8') if gzip_flag else cdn
                    fpt.write(cdn)


class CommandLineTextModel(CommandLineTest):
    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        logger = logging.getLogger('b4msa')
        logger.setLevel(self.data.verbose)
        with open(self.data.model, 'rb') as fpt:
            svc = pickle.load(fpt)
        with open(self.get_output(), 'w') as fpt:
            for tw in tweet_iterator(self.data.test_set):
                extra = dict([(int(a), float(b)) for a, b in svc.model[tw['text']]]
                             + [('num_terms', svc.num_terms)])
                tw.update(extra)
                fpt.write(json.dumps(tw) + "\n")


class CommandLineKfolds(CommandLineTrain):
    def __init__(self):
        super(CommandLineKfolds, self).__init__()
        self.param_kfold()

    def param_kfold(self):
        pa = self.parser.add_argument
        pa('--update-klass', default=False, dest='update_klass',
           action="store_true",
           help='Indicates whether the klass should be updated (default False)')
        pa('-k', '--kratio', dest='kratio',
           help='Predict the training set using k-fold (k > 1)',
           default="5",
           type=int)

    def main(self, args=None):
        self.data = self.parser.parse_args(args=args)
        assert not self.data.update_klass
        logging.basicConfig(level=self.data.verbose)
        logger = logging.getLogger('b4msa')
        logger.setLevel(self.data.verbose)
        best = load_json(self.data.params_fname)
        if isinstance(best, list):
            best = best[0]
        best = clean_params(best)
        print(self.data.params_fname, self.data.training_set)
        corpus, labels = read_data_labels(self.data.training_set)
        le = LabelEncoder()
        le.fit(labels)
        y = le.transform(labels)
        t = TextModel(corpus, **best)
        X = [t[x] for x in corpus]
        hy = [None for x in y]
        for tr, ts in KFold(n_splits=self.data.kratio,
                            shuffle=True, random_state=self.data.seed).split(X):
            c = SVC(model=t)
            c.fit([X[x] for x in tr], [y[x] for x in tr])
            _ = c.decision_function([X[x] for x in ts])
            [hy.__setitem__(k, v) for k, v in zip(ts, _)]

        i = 0
        with open(self.get_output(), 'w') as fpt:
            for tweet in tweet_iterator(self.data.training_set):
                tweet['decision_function'] = hy[i].tolist()
                i += 1
                fpt.write(json.dumps(tweet)+"\n")
        return hy


def params():
    c = CommandLine()
    c.main()


def train():
    c = CommandLineTrain()
    c.main()


def test():
    c = CommandLineTest()
    c.main()


def textmodel():
    c = CommandLineTextModel()
    c.main()


def kfolds(*args, **kwargs):
    c = CommandLineKfolds()
    return c.main(*args, **kwargs)
    
