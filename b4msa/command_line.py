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
from b4msa.utils import read_data_labels
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

    def param_search(self):
        pa = self.parser.add_argument
        pa('-s', '--sample', dest='samplesize', type=int,
           help="The sample size of the parameter space")
        pa('-q', '--qsize', dest='qsize', type=int, default=3,
           help="The minimum number of q-gram tokenizers per configuration")
        pa('-H', '--hillclimbing', dest='hill_climbing', default=False,
           action='store_true',
           help="Determines if hillclimbing search is also perfomed" +
           " to improve the selection of paramters")

    def param_set(self):
        pa = self.parser.add_argument
        pa('-n', '--numprocs', dest='numprocs', type=int, default=1,
           help="Number of processes to compute the best setup")
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
        if self.data.samplesize is not None:
            n_folds = self.data.n_folds
            n_folds = n_folds if n_folds is not None else 5
            perf, params = SVC.predict_kfold_params(self.data.training_set,
                                                    n_folds=n_folds,
                                                    n_params=self.data.samplesize,
                                                    seed=self.data.seed,
                                                    hill_climbing=self.data.hill_climbing,
                                                    qinitial=self.data.qsize,
                                                    numprocs=numprocs)
            params['score'] = perf
            with open(self.get_output(), 'w') as fpt:
                fpt.write(json.dumps(params, indent=2))
            return
        if self.data.n_folds is not None:
            pool = None if numprocs is None else Pool(numprocs)
            X, y = read_data_labels(self.data.training_set)
            hy = SVC.predict_kfold(X, y, n_folds=self.data.n_folds,
                                   seed=self.data.seed,
                                   pool=pool)
            if pool is not None:
                pool.close()
            with open(self.get_output(), 'w') as fpt:
                fpt.write("\n".join([str(x) for x in hy]))
            return


class CommandLineTrain(CommandLine):
    def __init__(self):
        super(CommandLineTrain, self).__init__()
        self.param_train()

    def param_search(self):
        pass

    def param_train(self):
        pa = self.parser.add_argument
        pa('-m', '--model-params', dest='params_fname', type=str,
           required=True,
           help="TextModel params")

    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        with open(self.data.params_fname) as fpt:
            params = json.loads(fpt.read())
        if 'score' in params:
            del params['score']
        svc = SVC.fit_from_file(self.data.training_set,
                                params)
        with open(self.get_output(), 'wb') as fpt:
            pickle.dump(svc, fpt)


def params():
    c = CommandLine()
    c.main()


def train():
    c = CommandLineTrain()
    c.main()
        
