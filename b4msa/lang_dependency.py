# Copyright 2016 Sabino Miranda-Jiménez
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import io
import re
import os
import logging
from nltk.stem.snowball import SnowballStemmer
from .params import OPTION_NONE
# from nltk.stem.porter import PorterStemmer
idModule = "language_dependency"
logger = logging.getLogger(idModule)
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatterC = logging.Formatter('%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
formatterC = logging.Formatter('%(module)s-%(funcName)s\n\t%(levelname)s\t%(message)s')
ch.setFormatter(formatterC)
logger.addHandler(ch)

PATH = os.path.join(os.path.dirname(__file__), 'resources')


_HASHTAG = '#'
_USERTAG = '@'
_sURL_TAG = '_url'
_sUSER_TAG = '_usr'
_sHASH_TAG = '_htag'
_sNUM_TAG = '_num'
_sDATE_TAG = '_date'
_sENTITY_TAG = '_ent'
_sNEGATIVE = "_neg"
_sPOSITIVE = "_pos"
_sNEUTRAL = "_neu"


class LangDependencyError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class LangDependency():
    """
    Defines a set of functions to change text using laguage dependent transformations, e.g., 
    - Negation
    - Stemming
    - Stopwords
    """
    STOPWORDS_CACHE = {}
    NEG_STOPWORDS_CACHE = {}

    def __init__(self, lang="spanish"):
        self.languages = ["spanish", "english", "italian", "german"]
        self.lang = lang

        if self.lang not in self.languages:
            raise LangDependencyError("Language not supported: " + lang)
        
        self.stopwords = LangDependency.STOPWORDS_CACHE.get(lang, None)
        if self.stopwords is None:
            self.stopwords = self.load_stopwords(os.path.join(PATH, "{0}.stopwords".format(lang)))
            LangDependency.STOPWORDS_CACHE[lang] = self.stopwords

        self.neg_stopwords = LangDependency.NEG_STOPWORDS_CACHE.get(lang, None)
        if self.neg_stopwords is None:
            self.neg_stopwords = self.load_stopwords(os.path.join(PATH, "{0}.neg.stopwords".format(lang)))
            LangDependency.NEG_STOPWORDS_CACHE[lang] = self.neg_stopwords

        if self.lang not in SnowballStemmer.languages:
            raise LangDependencyError("Unsupported language {0} (stemming)".format(lang))

        self.stemmer = SnowballStemmer(self.lang)

    def load_stopwords(self, fileName):
        """
         load stopwords from file
        """
        logger.debug("loading stopwords... " + fileName)
        if not os.path.isfile(fileName):
            raise LangDependencyError("File not found : " + fileName)                             
        
        StopWords = []
        with io.open(fileName, encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().lower()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                StopWords.append(line)

        return StopWords
                
    def stemming(self, text):
        """
        Applies the stemming process to text
        """
        
        tokens = re.split(r"~", text.strip())
        t = []
        for tok in tokens:
            if re.search(r"^(@|#|_|~)", tok, flags=re.I):
                t.append(tok)
            else:
                t.append(self.stemmer.stem(tok))
        return "~".join(t)

    def negation(self, text):
        """
        Applies negation process to the given text
        """
        if self.lang not in self.languages:
            raise LangDependencyError("Negation - language not defined")
        if self.lang == "spanish":
            text = self.spanish_negation(text)
        elif self.lang == "english":
            text = self.english_negation(text)
        elif self.lang == "italian":
            text = self.italian_negation(text)

        return text

    def spanish_negation(self, text):
        """
        Standarizes negation sentences, nouns are also considering with the operator "sin"
        "nunca jamás" is never changed
        """
        if getattr(self, 'pronouns', None) is None:
            self.pronouns = "me|te|se|lo|les|le|los"
            self.pronouns = self.pronouns + "|" + "|".join(self.neg_stopwords)

        text = text.replace('~', ' ')
        tags = _sURL_TAG + "|" + _sUSER_TAG + "|" + _sENTITY_TAG + "|" + \
               _sHASH_TAG + "|" + \
               _sNUM_TAG  + "|" + _sNEGATIVE + "|" + \
               _sPOSITIVE + "|" + _sNEUTRAL + "|"
  
        # reduces negation marks to a unique symbol
        text = re.sub(r"\b(jam[aá]s|nunca|sin|no)(\s+\1)+", r" \1 ", text, flags=re.I)

        p = re.compile(r"\b(nunca)\s+(?!jam[aá]s)")
        m = p.search(text)
        if m:
            text = p.sub(" no ", text)
        #
        text = re.sub(r"\b(jam[aá]s|nunca|sin|ni)\b", " no ", text, flags=re.I)
        text = re.sub(r"\b(jam[aá]s|nunca|sin|no)(\s+\1)+", r" \1 ", text, flags=re.I)
        # p1 = re.compile(r"(?P<neg>no)(?P<pron>(\s+(" +  pronombres + r"))*)\s+(?P<text>(?!("+ tags + ")(\s+|\b|$)))")
        p1 = re.compile(r"(?P<neg>((\s+|\b|^)no))(?P<pron>(\s+(" + self.pronouns + "|" + tags + r"))*)\s+(?P<text>(?!(" + tags + ")(\s+|\b|$)))", flags=re.I)
        m = p1.search(text)
        if m:
            text = p1.sub(r"\g<pron> \g<neg>_\g<text>", text)
        # remove isolated marks "no_" if marks appear because of negation rules
        text = re.sub(r"\b(no_)\b", r" no ", text, flags=re.I)
        text = re.sub(r"\s+", r" ", text, flags=re.I)
        return text.replace(' ', '~')

    def english_negation(self, text):
        """
        Standarizes negation sentences, nouns are also considering with the operator "without"
        Negative markers
        1. Not-negatior (not, n't)
        2. N-negator (never, neither, nobody, no, none, nor, nothing 
        3. Negative affix: this kind of negation is not dealt (-dis-confort, -a-symmetrical, -in-consistent) 
        """
        
        """

        VERBS
        not to VERB  => to no_VERB
        AUXn't VERB  => AUX no_VERB
        not VERB => no_VERB
        
        NOUNS
        
        no NOUN => no_NOUN
        
        ADJECTIVE

        BE_VERB not (prep) ADJ => BE_VERB prep no_ADJ 
       
        """        
        # pronouns = "me|you|he|she|it|us|them"

        return text

    def italian_negation(self, text):
        """
        Standarizes negation sentences, nouns are also considering with the operator "without"
        Negative markers
        1. Not-negatior (not, n't)
        2. N-negator (never, neither, nobody, no, none, nor, nothing 
        3. Negative affix: this kind of negation is not dealt (-dis-confort, -a-symmetrical, -in-consistent) 
        """
        
        """
        VERBS
        not to VERB  => to no_VERB
        AUXn't VERB  => AUX no_VERB
        not VERB => no_VERB
        
        NOUNS
        
        no NOUN => no_NOUN
        
        ADJECTIVE

        BE_VERB not (prep) ADJ => BE_VERB prep no_ADJ 
       
        """        
        #pronouns = "me|you|he|she|it|us|them"

        return text
    
    def filterStopWords(self, text, stopwords_option):
        if stopwords_option != 'none':
            for sw in self.stopwords:
                if stopwords_option == 'delete':
                    text = re.sub(r"\b(" + sw + r")\b", r"~", text, flags=re.I)
                elif stopwords_option == 'group':
                    text = re.sub(r"\b(" + sw + r")\b", r"~_sw~", text, flags=re.I)

        return text
    
    def transform(self, text, negation=False, stemming=False, stopwords=OPTION_NONE):
        if negation:
            text = self.negation(text)

        if stemming:
            text = self.stemming(text)

        text = self.filterStopWords(text, stopwords)
        return text
