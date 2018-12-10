# Copyright 2018 Mario Graff

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_space():
    from b4msa.textmodel import TextModel
    from b4msa.weighting import TFIDF
    from b4msa.utils import tweet_iterator
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(docs, token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = TFIDF(docs)
    assert len(sp.wordWeight) == len(sp._w2id)
    # print(sp._weight)
    # print(sp._w2id)


def test_doc2weight():
    from b4msa.textmodel import TextModel
    from b4msa.weighting import TFIDF
    from b4msa.utils import tweet_iterator
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(docs, token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = TFIDF(docs)
    assert len(sp.doc2weight(text.tokenize('odio odio los los'))) == 3


def test_getitem():
    from b4msa.textmodel import TextModel
    from b4msa.weighting import TFIDF
    from b4msa.utils import tweet_iterator
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(docs, token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = TFIDF(docs)
    tok = text.tokenize('buenos dias')
    bow = sp.doc2weight(tok)
    ids, tf, df = bow
    assert len(ids) == len(sp[tok])
