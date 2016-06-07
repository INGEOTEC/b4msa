[![Build Status](https://travis-ci.org/INGEOTEC/b4msa.svg?branch=master)](https://travis-ci.org/INGEOTEC/b4msa)

[![Coverage Status](https://coveralls.io/repos/github/INGEOTEC/b4msa/badge.svg?branch=master)](https://coveralls.io/github/INGEOTEC/b4msa?branch=master)

# A Baseline for Multilingual Sentiment Analysis (B4MSA) #

# Predict a training set using B4MSA #

Let us assume that B4MSA is installed and on the PATH, then the
stratisfied k-fold can be computed as follows:

```bash
b4msa -k 2 text.json
```

Note: it is recommended to install the following package

```bash
pip install tqdm
```

## Install B4MSA ##

Let us download python (from conda distribution), install it, and include python
in the PATH.

```bash   
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod 755 miniconda.sh
./miniconda.sh -b
export PATH=/home/$USER/miniconda3/bin:$PATH
```

B4MSA needs the following dependencies.

```bash
pip install coverage
pip install numpy
pip install scipy
pip install scikit-learn
pip install gensim
pip install nose
```

In order to install B4MSA use the following:   
* Clone the repository  
```bash   
git clone  https://github.com/INGEOTEC/b4msa.git
cd b4msa
```

* Install the package as usual  
```bash   
python setup.py install
```

* To install only for the use then  
```bash   
python setup.py install --user
```



