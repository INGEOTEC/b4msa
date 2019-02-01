.. EvoMSA documentation master file, created by
   sphinx-quickstart on Fri Aug  3 07:02:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

b4msa
==================================

.. image:: https://travis-ci.org/INGEOTEC/b4msa.svg?branch=master
	:target: https://travis-ci.org/INGEOTEC/b4msa   

.. image:: https://ci.appveyor.com/api/projects/status/y8vwd9998p74hw0n/branch/master?svg=true
	:target: https://ci.appveyor.com/project/mgraffg/b4msa/branch/master   

.. image:: https://coveralls.io/repos/github/INGEOTEC/b4msa/badge.svg?branch=master
	:target: https://coveralls.io/github/INGEOTEC/b4msa?branch=master

.. image:: https://anaconda.org/ingeotec/b4msa/badges/version.svg
	:target: https://anaconda.org/ingeotec/b4msa   

.. image:: https://badge.fury.io/py/b4msa.svg
	:target: https://badge.fury.io/py/b4msa

.. image:: https://readthedocs.org/projects/b4msa/badge/?version=latest		 
        :target: https://b4msa.readthedocs.io/en/latest/?badge=latest

b4msa is multilingual framework, that can be served as a baseline for sentiment analysis
classifiers, as well as a starting point to build new sentiment analysis
systems.

b4msa extends our work on creating a text classifier (see `microTC
<http://github.com/ingeotec/microtc>`_) by incorporating different
language dependent techniques such as:

* Stemming
* Stopword
* Negations

b4msa is described in `A Simple Approach to Multilingual Polarity
Classification in Twitter
<http://www.sciencedirect.com/science/article/pii/S0167865517301721>`_. 
Eric S. Tellez, Sabino Miranda-Jiménez, Mario Graff, Daniela
Moctezuma, Ranyart R. Suárez, Oscar S. Siordia. Pattern Recognition
Letters.

Citing
======

If you find b4msa useful for any academic/scientific purpose, we
would appreciate citations to the following reference:
  
.. code:: bibtex

	  @article{b4msa,
	title = {A {Simple} {Approach} to {Multilingual} {Polarity} {Classification} in {Twitter}},
	issn = {0167-8655},
	url = {http://www.sciencedirect.com/science/article/pii/S0167865517301721},
	doi = {10.1016/j.patrec.2017.05.024},
	abstract = {Recently, sentiment analysis has received a lot of attention due to the interest in mining opinions of social media users. Sentiment analysis consists in determining the polarity of a given text, i.e., its degree of positiveness or negativeness. Traditionally, Sentiment Analysis algorithms have been tailored to a specific language given the complexity of having a number of lexical variations and errors introduced by the people generating content. In this contribution, our aim is to provide a simple to implement and easy to use multilingual framework, that can serve as a baseline for sentiment analysis contests, and as a starting point to build new sentiment analysis systems. We compare our approach in eight different languages, three of them correspond to important international contests, namely, SemEval (English), TASS (Spanish), and SENTIPOLC (Italian). Within the competitions, our approach reaches from medium to high positions in the rankings; whereas in the remaining languages our approach outperforms the reported results.},
	urldate = {2017-05-24},
	journal = {Pattern Recognition Letters},
	author = {Tellez, Eric S. and Miranda-Jiménez, Sabino and Graff, Mario and Moctezuma, Daniela and Suárez, Ranyart R. and Siordia, Oscar S.},
	keywords = {Error-robust text representations, Multilingual sentiment analysis, Opinion mining},
	year = {2017}
	}

Installing b4msa
===============================

b4msa can be easly install using anaconda

.. code:: bash

	  conda install -c ingeotec b4msa

or can be install using pip, it depends on numpy, scipy and
scikit-learn.

.. code:: bash
	  
	  pip install numpy
	  pip install scipy
	  pip install scikit-learn
	  pip install microtc
	  pip install nltk
	  pip install b4msa

Text Model
=============

b4msa extends our work on creating a text classifier (specifically
:py:class:`microtc.textmodel.TextModel`) by incorporating different
language dependant techniques. 

.. autoclass:: b4msa.textmodel.TextModel
   :members:


Modules
==================	      

.. toctree::
   :maxdepth: 2

   classifier
   lang_dependency
