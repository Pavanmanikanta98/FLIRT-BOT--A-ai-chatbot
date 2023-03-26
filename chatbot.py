import random
import json
import pickle
import  os
import numpy as np
import requests
import nltk
import re
from nltk.stem import WordNetLemmatizer
import random
from tensorflow.keras.models import load_model

from random import randint
lemmatizer = WordNetLemmatizer()

#intents = text.loads(open('dialogues_act.txt').read())
with open('intents.json', 'r') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
Model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    #print(sentence_words)
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):

    bow = bag_of_words(sentence)
    res = Model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if 'response' in i:
                random_index = random.randint(0, len(i["response"]) - 1)
                return i['response'][random_index]
    return "I'm sorry, I don't understand."


#
# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             if 'response' in i:
#                 random_index =random.randint(0, len(i["response"]) - 1)
#     return intents['response'][random_index]




print("Leo Assistant is now ready for you...!")

while True:
    message = input(">>")
    if message.lower() == "byee":
       print("Goodbye!see you later .")
       break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)


