# Copyright 2016 Mario Graff (https://github.com/mgraffg) and Ranyart R. Suarez (https://github.com/RanyartRodrigo)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_SVC_predict_from_file():
    from b4msa.classifier import SVC
    from b4msa.textmodel import TextModel
    from b4msa.utils import read_data_labels
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    X, y = read_data_labels(fname)
    t = TextModel(X)
    c = SVC(t)
    c.fit_file(fname)
    y = c.predict_file(fname)
    for i in y:
        assert i in ['POS', 'NEU', 'NEG']


def test_SVC_predict():
    from b4msa.classifier import SVC
    from b4msa.textmodel import TextModel
    from b4msa.utils import read_data_labels
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    X, y = read_data_labels(fname)
    t = TextModel(X)
    c = SVC(t)
    c.fit_file(fname)
    y = c.predict_text('Excelente dia b4msa')
    assert y == 'POS'


def test_kfold():
    import os
    from b4msa.classifier import SVC
    from b4msa.utils import read_data_labels
    fname = os.path.dirname(__file__) + '/text.json'
    X, y = read_data_labels(fname, get_klass='klass',
                            get_tweet='text')
    hy = SVC.predict_kfold(X, y, n_folds=2)
    for x in hy:
        assert x in ['POS', 'NEU', 'NEG']


def test_kfold_pool():
    import os
    from b4msa.classifier import SVC
    from b4msa.utils import read_data_labels
    from multiprocessing import Pool
    fname = os.path.dirname(__file__) + '/text.json'
    X, y = read_data_labels(fname, get_klass='klass',
                            get_tweet='text')
    pool = Pool(2)
    hy = SVC.predict_kfold(X, y, n_folds=2, pool=pool)
    for x in hy:
        assert x in ['POS', 'NEU', 'NEG']
    pool.close()
    
