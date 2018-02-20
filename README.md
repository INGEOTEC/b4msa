[![Build Status](https://travis-ci.org/INGEOTEC/b4msa.svg?branch=master)](https://travis-ci.org/INGEOTEC/b4msa)
[![Build Status](https://travis-ci.org/INGEOTEC/b4msa.svg?branch=develop)](https://travis-ci.org/INGEOTEC/b4msa)
[![Build status](https://ci.appveyor.com/api/projects/status/y8vwd9998p74hw0n?svg=true)](https://ci.appveyor.com/project/mgraffg/b4msa)
[![Build status](https://ci.appveyor.com/api/projects/status/y8vwd9998p74hw0n/branch/master?svg=true)](https://ci.appveyor.com/project/mgraffg/b4msa/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/INGEOTEC/b4msa/badge.svg?branch=master)](https://coveralls.io/github/INGEOTEC/b4msa?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/b4msa/badges/version.svg)](https://anaconda.org/ingeotec/b4msa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/b4msa/badges/latest_release_date.svg)](https://anaconda.org/ingeotec/b4msa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/b4msa/badges/platforms.svg)](https://anaconda.org/ingeotec/b4msa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/b4msa/badges/installer/conda.svg)](https://conda.anaconda.org/ingeotec)
[![PyPI version](https://badge.fury.io/py/b4msa.svg)](https://badge.fury.io/py/b4msa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/b4msa/badges/license.svg)](https://anaconda.org/ingeotec/b4msa)

# A Baseline for Multilingual Sentiment Analysis (B4MSA)

B4MSA is a Python Sentiment Analysis Classifier for Twitter-like short texts. It can be used to create a first approximation to a sentiment classifier on any given language. It is almost language-independent, but it can take advantage of the particularities of a language.

It is written in Python making use of [NTLK](http://www.nltk.org/), [scikit-learn](http://scikit-learn.org/) and [gensim](https://radimrehurek.com/gensim/) to create simple but effective sentiment classifiers.

## Citing B4MSA ##

If you find B4MSA useful for any academic/scientific purpose, we would appreciate citations to the following journal paper: 

[A Simple Approach to Multilingual Polarity Classification in Twitter](http://www.sciencedirect.com/science/article/pii/S0167865517301721)
Eric S. Tellez, Sabino Miranda-Jiménez, Mario Graff, Daniela Moctezuma, Ranyart R. Suárez, Oscar S. Siordia. Pattern Recognition Letters.

```bibtex
@article{b4msa,
	title = {A {Simple} {Approach} to {Multilingual} {Polarity} {Classification} in {Twitter}},
	issn = {0167-8655},
	url = {http://www.sciencedirect.com/science/article/pii/S0167865517301721},
	doi = {10.1016/j.patrec.2017.05.024},
	abstract = {Recently, sentiment analysis has received a lot of attention due to the interest in mining opinions of social media users. Sentiment analysis consists in determining the polarity of a given text, i.e., its degree of positiveness or negativeness. Traditionally, Sentiment Analysis algorithms have been tailored to a specific language given the complexity of having a number of lexical variations and errors introduced by the people generating content. In this contribution, our aim is to provide a simple to implement and easy to use multilingual framework, that can serve as a baseline for sentiment analysis contests, and as a starting point to build new sentiment analysis systems. We compare our approach in eight different languages, three of them correspond to important international contests, namely, SemEval (English), TASS (Spanish), and SENTIPOLC (Italian). Within the competitions, our approach reaches from medium to high positions in the rankings; whereas in the remaining languages our approach outperforms the reported results.},
	urldate = {2017-05-24},
	journal = {Pattern Recognition Letters},
	author = {Tellez, Eric S. and Miranda-Jiménez, Sabino and Graff, Mario and Moctezuma, Daniela and Suárez, Ranyart R. and Siordia, Oscar S.},
	keywords = {Error-robust text representations, Multilingual sentiment analysis, Opinion mining}
}
```

A pre-print version can be found in [arXiv:1612.05270](https://arxiv.org/abs/1612.05270)

# Performance #

In order to know the performance of B4MSA, it was decided to compare
B4MSA against different commercial sentiment analyzers such as: [Google](http://cloud.google.com) 
[HP Haven](https://www.havenondemand.com), and
[Sentiment140](http://sentiment140.com). This comparison was performed
on **July 28, 2016**.

The performance
measures used in the comparison are taken from three different sentiment analysis competitions: TASS, 
SemEval, and SENTIPOLC. These measures are the accuracy, F1-positive-negative, and
F0F1-positive-negative. 

F1-positive-negative is defined as the average of the F1 obtained from
the positive class and F1 of the negative class.

F0F1-positive-negative is defined as the average between
F0-positive-negative and F1-positive-negative. F0-positive-negative is
defined as the average of the F0 of the positive class and F0 of the
negative class; F1-positive-negative is defined similarly. For more
information regarding this measure we refer the reader to the overview
of [SENTIPOLC challenge](http://www.fileli.unipi.it/projects/clic/proceedings/Proceedings-EVALITA-2014.pdf)

## SENTIPOLC 2014 (Italian)

|Algorithm | Accuracy | F1-positive-negative | F0F1-positive-negative |
|------|--------:|--------------:|---------------:|
|[HP Haven](https://www.havenondemand.com) | 0.4693 | 0.3921 | 0.4788 |
|B4MSA | **0.6072** | **0.6041** | **0.6354** |

## TASS 2015-small (Spanish)

|Algorithm | Accuracy | F1-positive-negative | F0F1-positive-negative |
|------|--------:|--------------:|---------------:|
| [HP Haven](https://www.havenondemand.com) |0.5350 | 0.4564 | 0.5775 |
|[Sentiment140](http://sentiment140.com) | 0.0590 | 0.1001 | 0.3225|
|[Meaning cloud](http://www.meaningcloud.com) |**0.6500** | 0.6640 | 0.6471|
| B4MSA | 0.6330 | **0.6949** | **0.6529** |

## SemEval 2015 (English)

|Algorithm | Accuracy | F1-positive-negative | F0F1-positive-negative |
|------|--------:|--------------:|---------------:|
|[HP Haven](https://www.havenondemand.com) | 0.6241 | **0.5724** | 0.6120 |
|[Google](http://cloud.google.com)  | 0.4218 | 0.4967 | 0.4624 |
|[Sentiment140](http://sentiment140.com) | 0.5553 | 0.4790 | 0.5368 |
|[Meaning cloud](http://www.meaningcloud.com) | 0.5775 | 0.5522 | 0.5361|
|B4MSA | **0.6281** | 0.5281 | **0.6578** |

## SemEval 2016 (English)

|Algorithm | Accuracy | F1-positive-negative | F0F1-positive-negative |
|------|--------:|--------------:|---------------:|
|[HP Haven](https://www.havenondemand.com) | **0.5986** | **0.5173** | **0.5337**  |
|[Google](http://cloud.google.com) | 0.3952 | 0.4728 | 0.4303 |
|[Sentiment140](http://sentiment140.com) | 0.5562 | 0.4135 | 0.4560|
|[Meaning cloud](http://www.meaningcloud.com) | 0.5252 | 0.5204 | 0.4861|
|B4MSA | 0.5106 | 0.4082 | 0.4866 |

### Note regarding [Google](http://cloud.google.com): ###

 [Google](http://cloud.google.com)'s sentiment analysis algorithm does
 not provide classes instead the service returns a continuous value
 between -1 and 1; however, in the api documentation it was mentioned
 the following rule:

```python
def get_class(pol):
    if pol < -0.75:
        return 'negative'
    elif pol > 0.25:
        return 'positive'
    return 'neutral'
```

# Installing B4MSA
B4MSA can be installed using [conda](https://www.continuum.io)  
```bash   
conda install -c ingeotec b4msa
```

B4MSA can be installed using `pip`
```bash
pip install b4msa
```
or cloning the [b4msa](https://github.com/INGEOTEC/b4msa) repository from github, e.g.,
```bash
git clone https://github.com/INGEOTEC/b4msa.git
```

# Random search on the B4MSA's parameters space #

Firstly, it is recommended to optimize the parameters used by
B4MSA. In order to free the user from this task, B4MSA can perform a random
search on the parameter space. B4MSA selects the best configuration found
on the random search. Suppose you have a workload of classified tweets
`tweets.json.gz` to model your problem, let us assume that b4msa is
already installed. In order to optimize B4MSA parameters one can use
the following command:

```bash
b4msa-params -k5 -s24 -n24 tweets.json.gz -o tweets.json
```

the parameters means for:

- `-k5` five folds
- `-s48` b4msa optimizes model's parameters for you, and `-s48` specifies that the parameter space should be sampled in 48 points and it simply get the best among them
- `-n24` let us specify the number of workds to be launch, it is a good idea to set `-s` as a multiply of `-n`.
- `-o tweets.json` specifies the file to store the configurations found by the parameter selection process, in best first order; a number of metrics are given,
  but it is in descending order by `_score`

The `tweets.json` looks like (for a four-classes problem)
```
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
```
each entry specifies a configuration, please check the code (a manual is coming soon) to learn about each parameter. Since first configurations show how best/good setups are composed, it is possible to learn something about your dataset making some analysis on these setups.

There exist other useful flags like:

- `-H` makes b4msa to perform last hill climbing search for the parameter selection, in many cases, this will produce much better configurations (never worst, guaranteed)
- `--lang spanish|english|italian` it specifies the language of the dataset, it allows b4msa to use language dependent techniques to the parameter selection procedure; currently, only `spanish` is supported.

```bash
b4msa-params -H -k5 -s48 -n24 tweets.json.gz -o tweets-spanish.json --lang spanish
```

The `tweets-spanish.json` file looks as follows:

```
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
```
Here we can see that `negation`, `stemming` and `stopwords` parameters were considered.

# Training the model to create a sentiment classifier

At this point, we are in the position to train a model. Let us that
the workload is `tweets.json.gz` and that the parameters are in
`tweets.json` then the following command will save the model
in `b4msa.model`

```bash
b4msa-train -o b4msa.model -m tweets.json tweets.json.gz
```

# Testing a sentiment classifier against a workload

At this point, we are in the position to test the model (i.e,
`b4msa.model`) in a new set. That is, we are in the position to ask
the classifier to assign a polarity label to a particular text. For
simplicity, let us assume that the new set is in `tweets.json.gz` and
the predicted labels are stored in `predicted.json.gz`.

```bash
b4msa-test -m b4msa.model -o predicted.json.gz tweets.json.gz
```

# Minimum requirements
In the modeling stage, the minimum requirements are dependent on the knowledge database being processed. Make sure you have enough memory for it. Take into account that b4msa can take advantage of multicore architectures using the `multiprocessing` module of python, this means that the memory requirements are multiplied by the number of processes you run.

It is recomended to use as many cores as you have to obtain good results in short running times.

On the training and testing stages only one core is used and there is no extra memory needs; however, no multicore support is provided for these stages.

# Installing dependencies

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
pip install nltk
```

For the eager people, it is recommended to install the `tqdm` package

```bash
pip install tqdm
```
However, it is better to prepare a coffee and a sandwich :)

