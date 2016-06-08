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
from nltk.stem.porter import PorterStemmer
idModule = "language_dependency"
logger = logging.getLogger(idModule)
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatterC = logging.Formatter('%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
formatterC = logging.Formatter('%(module)s-%(funcName)s\n\t%(levelname)s\t%(message)s')
ch.setFormatter(formatterC)
logger.addHandler(ch)

PATH = os.path.join(os.path.dirname(__file__), '../resources/')


_HASHTAG = '#'
_USERTAG = '@'
_sURL_TAG = 'url_tag'
_sUSER_TAG = 'user_tag'
_sHASH_TAG = 'hash_tag'
_sWINK_TAG = 'wink_tag'
_sNUM_TAG = 'num_tag'
_sDATE_TAG = 'date_tag'
_sENTITY_TAG = 'entity_tag'
_sNEGATIVE_TAG = "negativo_tag"
_sPOSITIVE_TAG = "positivo_tag"
_sNEUTRAL_TAG = "neutro_tag"
_sNEGATIVE_EMOTICON = "_negativo"
_sPOSITIVE_EMOTICON = "_positivo"
_sNEUTRAL_EMOTICON = "_neutro"


class LangDependencyError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class LangDependency():
    """
    Define a  set of functions to change text using laguage dependent transformations, e.g., 
    - Negation
    - Stemming
    - among others
    """

    def __init__(self, lang="spanish"):
        self.hshStopWords = {}
        self._sStopWords = ""
        self.hshNegationRules = None
        self.reNegationRules = None
        self.languages = ["spanish", "english", "italian", "german"]
        self.lang = lang

        if self.lang not in self.languages:
            raise LangDependencyError("Language not supported: " + lang)

        if self.lang not in SnowballStemmer.languages:
            raise LangDependencyError("Language stemming  not supported : " +
                                      lang)
        self.stemmer = SnowballStemmer(self.lang)

    def load_stopwords(self, fileName):
        """
         load stopwords from file
        """
        logger.debug("loading stopwords... " + fileName)
        self.hshStopWords = {}
        self._sStopWords = ""
        with io.open(fileName, encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().lower()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                self.hshStopWords[line] = line
                self._sStopWords = line + "|" + self._sStopWords
    
    def stemming(self, text):
        """
        Applies the stemming process to text
        """
        # logger.debug("stemming... ")
        
        tokens = re.split(r"\s+", text.strip())
        t = []
        for tok in tokens:
            if re.search(r"^(@|#)", tok, flags=re.I):
                t.append(tok)
            else:
                t.append(self.stemmer.stem(tok))
        return " ".join(t)

    def negation(self, text):
        """
        Applies negation process to the given text
        """
        if self.lang not in self.languages:
            raise LangDependencyError("Negation - language not defined")
        if self.lang == "spanish":
            text = self.spanish_negation(text)
        if self.lang == "english":
            text = self.english_negation(text)
        if self.lang == "italian":
            text = self.italian_negation(text)

        return text

    def spanish_negation(self, text):
        """
        Standarizes negation sentences, nouns are also considering with the operator "sin"
        "nunca jamás" is never changed
        """
        pronouns = "me|te|se|lo|les|le|los"
        pronouns = pronouns + "|" + self._sStopWords        
        tags = _sURL_TAG + "|" + _sUSER_TAG + "|" + _sENTITY_TAG + "|" + \
               _sWINK_TAG + "|" + _sHASH_TAG + "|" + \
               _sNUM_TAG  + "|" + _sNEGATIVE_TAG + "|" + \
               _sPOSITIVE_TAG + "|" + _sNEUTRAL_TAG + "|" + \
               _sPOSITIVE_EMOTICON + "|" + \
               _sNEGATIVE_EMOTICON + "|" + _sNEUTRAL_EMOTICON
 
        #reduces a unique negation mark
        text  = re.sub(r"\b(jam[aá]s|nunca|sin|no)(\s+\1)+", r"\1", text, flags=re.I)

        p = re.compile(r"\b(nunca)\s+(?!jam[aá]s)")
        m = p.search(text)
        if m:
            text = p.sub(" no ", text)
        #
        text = re.sub(r"\b(jam[aá]s|nunca|sin|ni)\b", " no ", text, flags=re.I)
        text = re.sub(r"\b(jam[aá]s|nunca|sin|no)(\s+\1)+", r"\1", text, flags=re.I)
        # p1 = re.compile(r"(?P<neg>no)(?P<pron>(\s+(" +  pronombres + r"))*)\s+(?P<text>(?!("+ tags + ")(\s+|\b|$)))")
        p1 = re.compile(r"(?P<neg>((\s+|\b|^)no))(?P<pron>(\s+(" + pronouns + "|" + tags + r"))*)\s+(?P<text>(?!(" + tags + ")(\s+|\b|$)))", flags=re.I) 
        m = p1.search(text)
        if m:
            text = p1.sub(r"\g<pron> \g<neg>_\g<text>", text)
        # remove isolated marks "no_" if marks appear because  negation rules
        text = re.sub(r"\b(no_)\b", r" no ", text, flags=re.I)
        return text

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
        #pronouns = "me|you|he|she|it|us|them"

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
    
    def filterStopWords(self, text):
        for sw in re.split(r"\s+", self._sStopWords):
            text = re.sub(r"\b(" + sw + r")\b", r" ", text, flags=re.I) 
        return text

