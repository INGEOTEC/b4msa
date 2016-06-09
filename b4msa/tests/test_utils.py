# author: Eric S. Tellez


def test_params():
    from b4msa.params import ParameterSelection
    import numpy as np
    sel = ParameterSelection()
    sel.search(lambda conf_code: (np.random.random(), conf_code[0]),
               bsize=64, qinitial=3)


def test_read_data_labels():
    import os
    from b4msa.utils import read_data_labels
    filename = os.path.join(os.path.dirname(__file__), "text.json")
    read_data_labels(filename)


