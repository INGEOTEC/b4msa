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

import xml.etree.ElementTree as ET

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s',
                    level=logging.INFO)


# Iterate over a file to get elements (tweets)...
def tweet_iterator(filename):

    # The file is a gz file (TODO: Test what is the content)...
    if filename.endswith(".gz"):
	# Uncompress the file and open it...
        f = gzip.GzipFile(filename)
        # Use JSON generator...
        return JSON_generator(f)
    # The file is an XML file...
    elif filename.endswith(".xml"):
	# Open the file...
        f = ET.parse(filename).getroot()
        # Use XML generator...
        return XML_generator(f)
    # JSON or unknown file format, guess...
    else:
	# Not JSON extension, show warning...
        if not filename.endswith(".json"):
            # Print warning to user...
            print("WARNING! File extension not supported (" + filename.split(".")[-1] + ")")
            print("Assuming JSON format: {\"text\":, \"klass\":}")
        # Open the file...
        f = open(filename, encoding='utf8')
	# Use JSON generator...
        return JSON_generator(f)

# Get tweets from an XML formatted file (e.g. TASS 2015)...
def XML_generator(f):

    # Start the iterator...
    for child in f:
	# Get the content...
        content = child.find('content').text
	# Get the class...
        klass = child.find('sentiments').find('polarity').find('value').text
	# Create an array compatible with JSON
        t = {'text': content, 'klass': klass}
	# Return the element... 
        yield t

	

# Get tweets from a JSON formated file (e.g. {"text":, "klass":})
def JSON_generator(f):

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
            print("WARNING! we found and error while parsing file:", filename)
            print("most of these errors occur due to concurrent writes")

    # Close the file...
    f.close()



def get_tweet(line):
    return json.loads(line)


def read_data_labels(filename, get_tweet='text', get_klass='klass', maxitems=1e100):
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


from xml.dom import minidom
def read_dataXML(filename, get_tweet='text', maxitems=1e100):
    data = []
    count = 0
    XML_File = open(filename, 'r')
    XML_Doc = minidom.parse(XML_File)


    for tweet in tweet_iterator(filename):
        count += 1
        x = get_tweet(tweet) if callable(get_tweet) else tweet[get_tweet]
        data.append(x)
        if count == maxitems:
            break

    return data

