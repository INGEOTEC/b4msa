|Build Status|

|Coverage Status|

A Baseline for Multilingual Sentiment Analysis (B4MSA)
======================================================

B4MSA is a Python Sentiment Analysis Classifier for Twitter-like short
texts. It can be used to create a first approximation to a sentiment
classifier on any given language. It is almost language-independent, but
it can take advantage of the particularities of a language.

It is written in Python making use of `NTLK <http://www.nltk.org/>`__,
`scikit-learn <http://scikit-learn.org/>`__ and
`gensim <https://radimrehurek.com/gensim/>`__ to create simple but
effective sentiment classifiers.

Installing B4MSA
================

B4MSA can be installed using ``pip``

.. code:: bash

    pip install b4msa

or cloning the `b4msa <https://github.com/INGEOTEC/b4msa>`__ repository
from github, e.g.,

.. code:: bash

    git clone https://github.com/INGEOTEC/b4msa.git

Predict a training set using B4MSA
==================================

Suppose you have a workload of classified tweets ``tweets.json.gz`` to
model your problem, let us assume that b4msa is already installed, then
the stratisfied k-fold can be computed as follows:

.. code:: bash

    b4msa-params -k5 -s24 -n24 tweets.json.gz -o tweets.json

the parameters means for:

-  ``-k5`` five folds
-  ``-s48`` b4msa optimizes model's parameters for you, and ``-s48``
   specifies that the parameter space should be sampled in 48 points and
   it simply get the best among them
-  ``-n24`` let us specify the number of workds to be launch, it is a
   good idea to set ``-s`` as a multiply of ``-n``.
-  ``-o tweets.json`` specifies the file to store the configurations
   found by the parameter selection process, in best first order; a
   number of metrics are given, but it is in descending order by
   ``_score``

The ``tweets.json`` looks like (for a four-classes problem)

::

    [
      {
        "_accuracy": 0.7773561997268175,
        "_macro_f1": 0.5703751933361809,
        "_score": 0.5703751933361809,
        "_time": 36.73965764045715,
        "_weighted_f1": 0.7467834129359526,
        "del_dup1": false,
        "lc": true,
        "num_option": "group",
        "strip_diac": true,
        "token_list": [
          1,
          2,
          3,
          6
        ],
        "url_option": "none",
        "usr_option": "group"
      },
    ...

each entry specifies a configuration, please check the code (a manual is
coming soon) to learn about each parameter. Since first configurations
show how best/good setups are composed, it is possible to learn
something about your dataset making some analysis on these setups.

There exist other useful flags like:

-  ``-H`` makes b4msa to perform last hill climbing search for the
   parameter selection, in many cases, this will produce much better
   configurations (never worst, guaranteed)
-  ``--lang spanish|english|german|italian`` it specifies the language
   of the dataset, it allows b4msa to use language dependent techniques
   to the parameter selection procedure; currently, only ``spanish`` is
   supported.

.. code:: bash

    b4msa-params -H -k5 -s48 -n24 tweets.json.gz -o tweets-spanish.json --lang spanish

The ``tweets-spanish.json`` file looks as follows:

::

    [
      {
        "_accuracy": 0.7750796782516315,
        "_macro_f1": 0.5736270120411987,
        "_score": 0.5736270120411987,
        "_time": 36.68731508255005,
        "_weighted_f1": 0.7472079134492694,
        "del_dup1": true,
        "lc": true,
        "negation": false,
        "num_option": "group",
        "stemming": true,
        "stopwords": "delete",
        "strip_diac": true,
        "token_list": [
          1,
          2,
          3,
          5
        ],
        "url_option": "delete",
        "usr_option": "none"
      },
    ...

Here we can see that ``negation``, ``stemming`` and ``stopwords``
parameters were considered.

Using the models to create a sentiment classifier
-------------------------------------------------

Testing a sentiment classifier against a workload
-------------------------------------------------

Minimum requirements
====================

In the modeling stage, the minimum requirements are dependent on the
knowledge database being processed. Make sure you have enough memory for
it. Take into account that b4msa can take advantage of multicore
architectures using the ``multiprocessing`` module of python, this means
that the memory requirements are multiplied by the number of processes
you run.

It is recomended to use as many cores as you have to obtain good results
in short running times.

On the training and testing stages only one core is used and there is no
extra memory needs; however, no multicore support is provided for these
stages.

Installing dependencies
=======================

Let us download python (from conda distribution), install it, and
include python in the PATH.

.. code:: bash

    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod 755 miniconda.sh
    ./miniconda.sh -b
    export PATH=/home/$USER/miniconda3/bin:$PATH

B4MSA needs the following dependencies.

.. code:: bash

    pip install coverage
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install gensim
    pip install nose
    pip install nltk

For the eager people, it is recommended to install the ``tqdm`` package

.. code:: bash

    pip install tqdm

However, it is better to prepare a coffee and a sandwich :)

.. |Build Status| image:: https://travis-ci.org/INGEOTEC/b4msa.svg?branch=master
   :target: https://travis-ci.org/INGEOTEC/b4msa
.. |Coverage Status| image:: https://coveralls.io/repos/github/INGEOTEC/b4msa/badge.svg?branch=master
   :target: https://coveralls.io/github/INGEOTEC/b4msa?branch=master
