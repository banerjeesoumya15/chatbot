import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

#import numpy as np
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# read and load the json file
with open('intents.json') as data_file:
    intents = json.load(data_file)

# Data Preprocessing
# tokenizing
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # add document to the corpus
        documents.append((w, intent['tag']))

        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])