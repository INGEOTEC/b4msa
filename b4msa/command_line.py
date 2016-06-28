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
from b4msa.classifier import SVC
from b4msa.utils import read_data_labels, read_data
# from b4msa.params import OPTION_DELETE
from multiprocessing import Pool, cpu_count
import json
import pickle

# from b4msa.params import ParameterSelection


class CommandLine(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.training_set()
        self.predict_kfold()
        self.param_set()
        self.param_search()
        self.langdep()

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
           help='Logging level default: INFO + 1',
           default=logging.INFO+1)

    def langdep(self):
        pa = self.parser.add_argument
        pa('-l', '--lang', dest='lang', type=str, default=None,
           help="the language")

    def param_search(self):
        pa = self.parser.add_argument
        pa('-s', '--sample', dest='samplesize', type=int,
           default=8,
           help="The sample size of the parameter")
        pa('-q', '--qsize', dest='qsize', type=int, default=3,
           help="The size of the initial population of tokenizers")
        pa('-H', '--hillclimbing', dest='hill_climbing', default=False,
           action='store_true',
           help="Determines if hillclimbing search is also perfomed to improve the selection of paramters")
        pa('-n', '--numprocs', dest='numprocs', type=int, default=1,
           help="Number of processes to compute the best setup")
        pa('-S', '--score', dest='score', type=str, default='macrof1',
           help="The name of the score to be optimized (macrof1|weightedf1|accuracy|avgf1:klass1:klass2); it defaults to macrof1")

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

        if self.data.numprocs == 1:
            numprocs = None
        elif self.data.numprocs == 0:
            numprocs = cpu_count()
        else:
            numprocs = self.data.numprocs

        n_folds = self.data.n_folds
        n_folds = n_folds if n_folds is not None else 5
        assert self.data.score.split(":")[0] in ('macrof1', 'microf1', 'weightedf1', 'accuracy', 'avgf1'), "Unknown score {0}".format(self.data.score)

        best_list = SVC.predict_kfold_params(
            self.data.training_set,
            n_folds=n_folds,
            score=self.data.score,
            numprocs=numprocs,
            seed=self.data.seed,
            param_kwargs=dict(
                bsize=self.data.samplesize,
                hill_climbing=self.data.hill_climbing,
                qsize=self.data.qsize,
                lang=self.data.lang
            )
        )
        with open(self.get_output(), 'w') as fpt:
            fpt.write(json.dumps(best_list, indent=2, sort_keys=True))


class CommandLineTrain(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.param_set()
        self.training_set()
        self.param_train()

    def param_train(self):
        pa = self.parser.add_argument
        pa('-m', '--model-params', dest='params_fname', type=str,
           required=True,
           help="TextModel params")

    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        with open(self.data.params_fname) as fpt:
            param_list = json.loads(fpt.read())

        best = param_list[0]
        svc = SVC.fit_from_file(self.data.training_set, best)
        
        with open(self.get_output(), 'wb') as fpt:
            pickle.dump(svc, fpt)


class CommandLineTest(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.param_set()
        self.training_set()
        self.param_test()

    def param_test(self):
        pa = self.parser.add_argument
        pa('-m', '--model', dest='model', type=str,
           required=True,
           help="SVM Model file name")

    def training_set(self):
        cdn = 'File containing the test set'
        pa = self.parser.add_argument
        pa('test_set',
           default=None,
           help=cdn)
        pa('--verbose', dest='verbose', type=int,
           help='Logging level default: INFO + 1',
           default=logging.INFO+1)

    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        with open(self.data.model, 'rb') as fpt:
            svc = pickle.load(fpt)
        X = [svc.model[x] for x in read_data(self.data.test_set)]
        hy = svc.predict(X)
        with open(self.get_output(), 'w') as fpt:
            # fpt.write("\n".join([str(x) for x in hy]))
            for text, klass in zip(read_data(self.data.test_set), hy):
                fpt.write(json.dumps({"text": text, "klass": klass})+"\n")


class CommandLineTextModel(CommandLineTest):
    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        with open(self.data.model, 'rb') as fpt:
            svc = pickle.load(fpt)
        X = [svc.model[x] for x in read_data(self.data.test_set)]
        with open(self.get_output(), 'w') as fpt:
            # fpt.write("\n".join([str(x) for x in hy]))
            for x in X:
                fpt.write(json.dumps(dict(x + [('num_terms', svc.num_terms)]))+"\n")
                    

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
