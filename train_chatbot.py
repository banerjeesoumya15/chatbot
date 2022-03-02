import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
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