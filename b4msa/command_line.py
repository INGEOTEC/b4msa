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
from b4msa.classifier import SVC
from multiprocessing import cpu_count, Pool

# from b4msa.params import ParameterSelection


class CommandLine(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='b4msa')
        self.training_set()
        self.predict_kfold()
        self.param_set()

    def predict_kfold(self):
        pa = self.parser.add_argument
        pa('-k', '--kfolds', dest='n_folds',
           help='Predict the training set using stratified k-fold', type=int, default=5)

    def training_set(self):
        cdn = 'File containing the training set on csv.'
        self.parser.add_argument('training_set',
                                 nargs=1,  # '?'
                                 default=None,
                                 help=cdn)

    def param_set(self):
        pa = self.parser.add_argument
        pa('-s', '--sample', dest='samplesize', type=int, default=1,
           help="The sample size of the parameter space")
        pa('-q', '--qsize', dest='qsize', type=int, default=3,
           help="The minimum number of q-gram tokenizers per configuration")
        pa('-n', '--numprocs', dest='numprocs', type=int, default=1,
           help="Number of processes to compute the best setup")
        pa('-H', '--hillclimbing', dest='hill_climbing', default=False, action='store_true',
           help="Determines if hillclimbing search is also perfomed to improve the selection of paramters")

    def main(self):
        self.data = self.parser.parse_args()
        # if self.data.n_folds is not None and self.data.sample is not None:

        if self.data.numprocs == 1:
            pool = None
        elif self.data.numprocs == 0:
            pool = Pool(cpu_count())
        else:
            pool = Pool(self.data.numprocs)

        for filename in self.data.training_set:
            hy = SVC.predict_kfold_params(filename,
                                          n_folds=self.data.n_folds,
                                          n_params=self.data.samplesize,
                                          hill_climbing=self.data.hill_climbing,
                                          qinitial=self.data.qsize,
                                          pool=pool
            )
            print("filename: {0}; score: {1}".format(filename, hy))
        # elif self.data.n_folds is not None:
        #  hy = SVC.predict_kfold(self.data.training_set,
        #                         n_folds=self.data.n_folds)
        #  print(hy)


def main():
    c = CommandLine()
    c.main()
    
