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
import numpy as np


def test_nparams():
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-k', '2', '-s', '11', fname]
    c.main()
    os.unlink(c.get_output())


def test_main():
    from b4msa.command_line import params
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname]
    params()
    os.unlink(output)


def test_pool():
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-k', '2', '-s', '11', '-n', '2', fname]
    c.main()
    os.unlink(c.get_output())


def test_output():
    from b4msa.command_line import CommandLine
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname]
    c.main()
    assert os.path.isfile(output)
    os.unlink(output)


def test_seed():
    try:
        from mock import MagicMock
    except ImportError:
        from unittest.mock import MagicMock
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    seed = np.random.seed
    np.random.seed = MagicMock()
    c = CommandLine()
    sys.argv = ['b4msa', '-s', '2', '--seed', '1', '-k', '2', fname]
    c.main()
    os.unlink(c.get_output())
    np.random.seed.assert_called_once_with(1)
    np.random.seed = seed


def test_train():
    from b4msa.command_line import CommandLine, CommandLineTrain
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    c.main()
    assert os.path.isfile(output)
    with open(output) as fpt:
        print(fpt.read())
    c = CommandLineTrain()
    sys.argv = ['b4msa', '-m', output, fname]
    print(c.main())
    os.unlink(output)
    os.unlink(c.get_output())
        

def test_train2():
    from b4msa.command_line import CommandLine, train
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    c.main()
    assert os.path.isfile(output)
    output2 = tempfile.mktemp()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output2]
    train()
    os.unlink(output)
    os.unlink(output2)


def test_test():
    from b4msa.command_line import params, train, test
    from b4msa.utils import read_data_labels
    import os
    import sys
    import tempfile
    import json
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output]
    train()
    output2 = tempfile.mktemp()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output2]
    test()
    X, y = read_data_labels(output2)
    print(y)
    os.unlink(output)
    with open(output2) as fpt:
        a = [json.loads(x) for x in fpt.readlines()]
    os.unlink(output2)
    for x in a:
        assert 'q_voc_ratio' in x
    assert len(y)


def test_decision_function():
    from b4msa.command_line import params, train, test
    from b4msa.utils import tweet_iterator
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output]
    train()
    output2 = tempfile.mktemp()
    sys.argv = ['b4msa', '-m', output, fname,
                '-o', output2, '--decision-function']
    test()
    d = [x for x in tweet_iterator(output2)]
    os.unlink(output)
    os.unlink(output2)
    assert len(d)
    assert len(d) == len([x for x in d if 'decision_function' in x])


def test_score():
    from b4msa.command_line import params
    import os
    import sys
    import tempfile
    import json
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2', '-S', 'avgf1:POS:NEG']
    params()
    with open(output) as fpt:
        a = json.loads(fpt.read())[0]
    assert a['_score'] == a['_avgf1:POS:NEG']
    os.unlink(output)
        

def test_textmodel():
    from b4msa.command_line import params, train, textmodel
    import os
    import sys
    import json
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output]
    train()
    output2 = tempfile.mktemp()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output2]
    textmodel()
    os.unlink(output)
    a = open(output2).readline()
    os.unlink(output2)
    a = json.loads(a)
    assert 'klass' in a


def test_params_gzip():
    from b4msa.command_line import params
    import os
    import sys
    import tempfile
    import json
    import gzip
    output = tempfile.mktemp() + '.gz'
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2', '-S', 'avgf1:POS:NEG']
    params()
    with gzip.open(output) as fpt:
        b = fpt.read()
        a = json.loads(str(b, encoding='utf-8'))[0]
    assert a['_score'] == a['_avgf1:POS:NEG']
    os.unlink(output)
    

def test_params_gzip2():
    from b4msa.command_line import params, train
    import os
    import sys
    import tempfile
    output = tempfile.mktemp() + '.gz'
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output]
    train()


def test_decision_function_gzip():
    from b4msa.command_line import params, train, test
    from b4msa.utils import tweet_iterator
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-H', '-lspanish', '-o', output,
                '-k', '2', fname, '-s', '2', '-n0']
    params()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output]
    train()
    output2 = tempfile.mktemp() + '.gz'
    sys.argv = ['b4msa', '-m', output, fname,
                '-o', output2, '--decision-function']
    test()
    d = [x for x in tweet_iterator(output2)]
    os.unlink(output)
    os.unlink(output2)
    assert len(d)
    assert len(d) == len([x for x in d if 'decision_function' in x])


def test_kfolds():
    from b4msa.command_line import params, kfolds
    import os
    import sys
    import json
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    output2 = tempfile.mktemp()
    sys.argv = ['b4msa', '-m', output, fname, '-o', output2]
    print(output, fname)
    kfolds()
    os.unlink(output)
    a = open(output2).readline()
    os.unlink(output2)
    a = json.loads(a)
    assert 'decision_function' in a
    sys.argv = ['b4msa', '--update-klass', '-m', output, fname, '-o', output2]
    try:
        kfolds()
    except AssertionError:
        return
    assert False
    
