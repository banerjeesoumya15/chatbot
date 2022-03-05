import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

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

# create training data
training = []
#output_row = [0] * len(classes)
for doc in documents:
    bow = []
    #pattern_words = doc[0]

    # lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    # create bag of words
    for w in words:
        bow.append(1) if w in pattern_words else bow.append(0)

    # output is a 0 for each tag and 1 for current tag (for each patern)
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1

    training.append([bow, output_row])

# shuffle and change to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# create train set
# X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Train data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=10, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model saved")