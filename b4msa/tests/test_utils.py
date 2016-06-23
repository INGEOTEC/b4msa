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
