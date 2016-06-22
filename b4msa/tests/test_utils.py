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


def test_pos_neg_f1():
    import os
    from b4msa.utils import read_data_labels, pos_neg_f1
    from sklearn.metrics import f1_score
    from sklearn import preprocessing
    import numpy as np
    filename = os.path.join(os.path.dirname(__file__), "text.json")
    X, y = read_data_labels(filename)
    le = preprocessing.LabelEncoder().fit(y)
    y = np.array(le.transform(y))
    hy = y.copy()
    np.random.seed(0)
    np.random.shuffle(hy)
    assert pos_neg_f1(y, hy) == f1_score(y, hy, average=None)[:2].mean()
