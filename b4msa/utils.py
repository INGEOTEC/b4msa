# Copyright 2016 Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import gzip
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s',
                    level=logging.INFO)


def tweet_iterator(filename):
    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename)
    else:
        f = open(filename, encoding='ascii')

    while True:
        line = f.readline()
        if type(line) is bytes:
            line = str(line, encoding='ascii')
        if len(line) == 0:
            break

        line = line.strip()
        if len(line) == 0:
            continue

        # print(line)
        t = None
        try:
            t = get_tweet(line)
            yield t
        except (json.decoder.JSONDecodeError, ValueError):
            print("WARNING! we found and error while parsing file:", filename)
            print("most of these errors occur due to concurrent writes")

    f.close()


def get_tweet(line):
    return json.loads(line)


def read_data_labels(filename, maxitems=1e100,
                     get_tweet='text', get_klass='klass'):
    data, labels = [], []
    count = 0
    for tweet in tweet_iterator(filename):
        count += 1
        x = get_tweet(tweet) if callable(get_tweet) else tweet[get_tweet]
        y = get_klass(tweet) if callable(get_klass) else tweet[get_klass]
        data.append(x)
        labels.append(y)
        if count == maxitems:
            break
    return data, labels

