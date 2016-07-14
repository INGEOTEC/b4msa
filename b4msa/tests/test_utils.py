# author: Eric S. Tellez


def test_params():
    from b4msa.params import ParameterSelection
    import numpy as np
    sel = ParameterSelection()
    
    def fake_score(conf_code):
        conf, code = conf_code
        conf['_score'] = np.random.random()
        conf['_time'] = 1.0
        return conf
        
    sel.search(fake_score, bsize=64, qsize=3)


def test_read_data_labels():
    import os
    from b4msa.utils import read_data_labels
    filename = os.path.join(os.path.dirname(__file__), "text.json")
    read_data_labels(filename)


def test_wrapper_score():
    from b4msa.params import Wrapper
    from sklearn.metrics import f1_score, precision_score
    import numpy as np
    np.random.seed(0)
    y = np.random.randint(3, size=100).astype(np.str)
    hy = np.random.randint(3, size=100)
    w = Wrapper(None, y, 'avgf1:0:2', 10, None)
    conf = {}
    w._score(conf, hy)
    f1 = f1_score(y.astype(np.int), hy, average=None)
    assert conf['_accuracy'] == (y.astype(np.int) == hy).mean()
    print(conf['_avgf1:0:2'], (f1[0] + f1[2]) / 2.)
    assert conf['_avgf1:0:2'] == (f1[0] + f1[2]) / 2.
    precision = precision_score(y.astype(np.int), hy, average=None)
    pos = (f1[0] + precision[0]) / 2.
    neg = (f1[2] + precision[2]) / 2.
    w = Wrapper(None, y, 'avgf1f0:0:2', 10, None)
    w._score(conf, hy)
    assert conf['_avgf1f0:0:2'] == (pos + neg) / 2.
