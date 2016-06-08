# author: Eric S. Tellez
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s',
                    level=logging.INFO)


def test_params():
    from b4msa.params import ParameterSelection
    import numpy as np
    sel = ParameterSelection()
    sel.search(lambda conf_code: np.random.random(), bsize=64, qinitial=3)


def test_read_data_labels():
    import os
    from b4msa.utils import read_data_labels
    filename = os.path.join(os.path.dirname(__file__), "text.json")
    read_data_labels(filename)


