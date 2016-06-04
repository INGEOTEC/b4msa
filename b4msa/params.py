import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s',
                    level=logging.INFO)

OPTION_NONE = 'none'
OPTION_GROUP = 'group'
OPTION_DELETE = 'delete'


basic_options = [OPTION_DELETE, OPTION_GROUP, OPTION_NONE]
base_params = dict(
    strip_diac=[False, True],
    usr_option=basic_options,
    url_option=basic_options,
    lc=[False, True],
    token_list=[1, 2, 3, 4, 5, 6, 7],
)

_base_params = sorted(base_params.items())


class ParameterSelection:
    def __init__(self):
        pass

    def sample_param_space(self, n, q=3):
        for i in range(n):
            kwargs = {}
            for k, v in _base_params:
                if len(v) == 0:
                    continue

                if k == 'token_list':
                    x = list(v)
                    np.random.shuffle(x)
                    kwargs[k] = sorted(x[:q])
                else:
                    print(v)
                    kwargs[k] = np.random.choice(v)

            yield kwargs

    def expand_neighbors(self, s):
        for k, v in s.items():
            if v in (True, False):
                x = s.copy()
                x[k] = not v
                yield x
            elif v in basic_options:
                for _v in basic_options:
                    if _v != v:
                        x = s.copy()
                        x[k] = _v
                        yield x
            elif k == 'token_list':
                for _v in base_params[k]:
                    if _v not in v:
                        x = s.copy()
                        x[k] = x[k].copy()
                        x[k].append(_v)
                        yield x

    def search(self, fun_score, bsize=32, qinitial=3):
        tabu = set()  # memory for tabu search
        best = (0, None)
        # initial approximation, montecarlo based process
        for conf in self.sample_param_space(bsize, q=qinitial):
            code = get_filename(conf)
            if code in tabu:
                continue

            tabu.add(code)
            best = max(best, (fun_score(conf), conf))

        # second approximation, hill climbing process
        while True:
            bscore = best[0]
            for conf in self.expand_neighbors(best[1]):
                code = get_filename(conf)
                if code in tabu:
                    continue

                tabu.add(code)
                best = max(best, (fun_score(conf), conf))

            if bscore == best[0]:
                break

        return best

def get_filename(kwargs, basename=None):
    L = []
    if basename:
        L.append(basename)
        
    for k, v in sorted(kwargs.items()):
        L.append("{0}={1}".format(k, v).replace(" ", ""))

    return "-".join(L)

