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

# lemmatize, change to lower case, remove duplicates
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))  # sort classes
print("Length of documents: ", len(documents))  # documents is combination of patterns and intents
print("Length of classes: ", len(classes))
print("Length of lemmatized words: ", len(words))

# store lemmatized words
# store classes/intents
with open('words.pkl', 'wb') as lem_file:
    pickle.dump(words, lem_file)
with open('classes.pkl', 'wb') as classes_file:
    pickle.dump(classes, classes_file)