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


def test_lang_dependency():
    from b4msa.lang_dependency import LangDependency
    LangDependency()


def test_spanish_stemming():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='spanish')
    r = c.stemming('los carros son veloces')
    assert r == 'los carr son veloc'
        

def test_spanish_negation():
    from b4msa.lang_dependency import LangDependency
    c = LangDependency(lang='spanish')
    r = c.negation('los carros no son veloces')
    assert r == 'los carros  no_son veloces'
