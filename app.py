import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np

from tensorflow.keras.models import load_model
import json
import random

# load model
MODEL_FILE = 'chatbot_model.h5'
ERROR_THRESHOLD = 0.0  # define error threshold level
model = load_model(MODEL_FILE)
# read and load the json file
with open('intents.json') as data_file:
    intents = json.load(data_file)
# load lemmatized words
with open('words.pkl', 'rb') as words_file:
    words = pickle.load(words_file)
# load classes
with open('classes.pkl', 'rb') as classes_file:
    classes = pickle.load(classes_file)

# preprocessing
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# bag of words
def bag_of_words(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bow = [0] * len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bow[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bow)

def predict_class(sentence, model):
    # filter out predictions below threshold
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]],
                            "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Creating GUI with tkinter
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + "\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + "\n\n")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# create chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")

ChatLog.config(state=DISABLED)

# Bind scrollbar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# create button to send message
SendButton = Button(base, font=("Verdana", 12, "bold"),
                    text="Send",
                    width="12",
                    height=5,
                    bd=0,
                    bg="#32de97",
                    activebackground="#3c9d9b",
                    fg="#ffffff",
                    command=send)
# create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

# Placement in screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
