# author: Eric S. Tellez
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s',
                    level=logging.INFO)

def test_params():
    from b4msa.params import sample_param_space, expand_neighbors
    n = 64
    x = list(sample_param_space(n, 3))
    for _x in x:
        logging.info("============= {0} ==========".format(_x))
        for y in expand_neighbors(_x):
            logging.info(y)


def test_read_data_labels():
    import os
    from b4msa.utils import read_data_labels
    filename = os.path.join(os.path.dirname(__file__), "text.json")
    read_data_labels(filename)


