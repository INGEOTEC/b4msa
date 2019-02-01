# Copyright 2016 Sabino Miranda-Jiménez and Mario Graff (https://github.com/mgraffg) 
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


def test_spanish_stemming():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='spanish')
    r = c.stemming('los~carros~son~veloces')
    assert r.split('~') == 'los carr son veloc'.split()
        

def test_spanish_negation():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='spanish')
    r = c.negation('los carros no son veloces')
    print(r)
    assert r.split('~') == 'los carros no_son veloces'.split()


def test_arabic_stemming():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='arabic')
    r = c.stemming('los~carros~no~son~veloces')
    print(r)


def test_stopwords_property():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='spanish')
    print(len(c.stopwords))
    assert len(c.stopwords) == 12
    c = LangDependency(lang='arabic')
    print(len(c.stopwords))
    assert len(c.stopwords) == 9


def test_neg_stopwords_property():
    from b4msa.lang_dependency import LangDependency, LangDependencyError
    c = LangDependency(lang='spanish')
    print(len(c.neg_stopwords))
    assert len(c.neg_stopwords) == 180
    c = LangDependency(lang='arabic')
    try:
        print(c.neg_stopwords)
    except LangDependencyError:
        return
    assert False


def test_lang_abbr():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='es')
    assert c.lang == 'spanish'


def test_stopwords():
    from b4msa.textmodel import TextModel
    tm = TextModel(lang='es', del_dup=False)
    text = tm.text_transformations('como esta mi carro')
    print(text)
    text1 = tm.lang.transform(text, stopwords='delete')
    print(text1)
    assert text1 == '~carro~'
    text1 = tm.lang.transform(text, stopwords='group')
    print(text1)
    assert text1 == '~_sw~_sw~_sw~carro~'
