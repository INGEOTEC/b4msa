|Build Status|

|Coverage Status|

A Baseline for Multilingual Sentiment Analysis (B4MSA)
======================================================

Predict a training set using B4MSA
==================================

Let us assume that B4MSA is installed only for a particular user.

.. code:: bash

    ~/.local/bin/b4msa -k 2 text.json

Install B4MSA
-------------

Let us download python (from conda distribution), install it, and
include python in the PATH.

.. code:: bash

    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod 755 miniconda.sh
    ./miniconda.sh -b
    export PATH=/home/travis/miniconda3/bin:$PATH

B4MSA needs the following dependencies.

.. code:: bash

    pip install coverage
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install gensim
    pip install nose

In order to install B4MSA use the following: \* Clone the repository

.. code:: bash

    git clone  https://github.com/INGEOTEC/b4msa.git
    cd b4msa

-  Install the package as usual

   .. code:: bash

       python setup.py install

-  To install only for the use then

   .. code:: bash

       python setup.py install --user

.. |Build Status| image:: https://travis-ci.org/INGEOTEC/b4msa.svg?branch=master
   :target: https://travis-ci.org/INGEOTEC/b4msa
.. |Coverage Status| image:: https://coveralls.io/repos/github/INGEOTEC/b4msa/badge.svg?branch=master
   :target: https://coveralls.io/github/INGEOTEC/b4msa?branch=master
