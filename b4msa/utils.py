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

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


def tweet_iterator(filename):

    # The file is a gz file...
    if filename.endswith(".gz"):
        # Uncompress the file and open it...
        f = gzip.GzipFile(filename)
    else:
        # Not JSON extension, show warning...
        if not filename.endswith(".json"):
            # Print warning to user...
            print("WARNING! File extension not supported (" + filename.split(".")[-1] + ")")
            print("Assuming JSON format: {\"text\":, \"klass\":}")
        # Open the file...
        f = open(filename, encoding='utf8')

    # Start the iterator...
    while True:
        # Get the nex line...
        line = f.readline()
        # Test the type of the line and encode it if neccesary...
        if type(line) is bytes:
            line = str(line, encoding='utf8')
        # If the line is empty, we are done...
        if len(line) == 0:
            break
        # Remove whitespaces from the beginning and the end...
        line = line.strip()
        # If line is empty, jump to next...
        if len(line) == 0:
            continue
        # Clean var to yield...
        t = None

        try:
            # Try to get data from JSON format...
            t = get_tweet(line)
            # Return the generator...
            yield t
        # Catch the JSON Decode Error...
        except (json.decoder.JSONDecodeError, ValueError):
            # Print warning to user...
            print("WARNING! we found and error while parsing file:", str(f))
            print("most of these errors occur due to concurrent writes")

    # Close the file...
    f.close()


def get_tweet(line):
    return json.loads(line)


def read_data_labels(filename, get_tweet='text',
                     get_klass='klass', maxitems=1e100):
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


def read_data(filename, get_tweet='text', maxitems=1e100):
    data = []
    count = 0
    for tweet in tweet_iterator(filename):
        count += 1
        x = get_tweet(tweet) if callable(get_tweet) else tweet[get_tweet]
        data.append(x)
        if count == maxitems:
            break

    return data

