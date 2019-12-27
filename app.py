import logging
logging.getLogger('tensorflow').disabled = True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#import tensorflow as tf


from gtts import gTTS
import speech_recognition as sr
import os
import re
import webbrowser
import smtplib
import requests
import bs4
from weather import Weather
#import pyttsx3
import random
import sys
import wikipedia
import json
import pprint
import datetime

now = datetime.datetime.now()

def talkToMe(audio):

    print(audio)
    for line in audio.splitlines():
        os.system("say " + audio)

def weather_data(query):
    res=requests.get('http://api.openweathermap.org/data/2.5/weather?'+query+'&APPID=a946124530bfdabc290682b3c1131b47&units=metric');
    return res.json();
def print_weather(result,city):
    print("{}'s temperature: {}°C ".format(city,result['main']['temp']))
    print("Wind speed: {} m/s".format(result['wind']['speed']))
    print("Description: {}".format(result['weather'][0]['description']))
    print("Weather: {}".format(result['weather'][0]['main']))

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

words = []
labels = []
docs_x = []
docs_y = []
with open("intents.json") as file:
    data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#try:
#    model.load("model.tflearn")
#except:
#    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#    model.save("model.tflearn")

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    talkToMe('Hi, I am Jarvis Assistant')
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        elif "wiki" in inp.lower():
            command =inp.lower().replace('wiki','')
            results = wikipedia.summary(command, sentences=2)
            talkToMe('According to wikipedia ')
            print(results)
            continue
        elif 'open website' in inp.lower():
            reg_ex = re.search('open website (.+)', inp.lower())
            if reg_ex:
                domain = reg_ex.group(1)
                url = 'https://www.' + domain+'.com'
                webbrowser.open(url)
                print('Done!')
                continue
            else:
                pass
        elif 'what\'s up' in inp.lower():
            print('Just doing my thing')
            continue

        elif 'weather' in inp.lower():
            city=input('Enter the city:')
            query='q='+city;
            w_data=weather_data(query);
            print_weather(w_data, city)
            continue
        elif 'sebok' in inp.lower():
            #print('Opening sebokwiki for your search query')
            talkToMe('Opening sebokwiki for your search query')
            command =inp.lower().replace('sebok','')
            url = 'https://www.sebokwiki.org/w/index.php?search=' + command
            webbrowser.open(url)
            #print(command.xpath("//div[@class='r']").xpath("//a/@href").extract())
            print('Done!')
            continue
        elif 'youtube' in inp.lower():
            #print('Opening youtube for your search query')
            talkToMe('Opening youtube for your search query')
            command =inp.lower().replace('youtube','')
            url = 'https://www.youtube.com/results?search_query=' + command
            webbrowser.open(url)
            print('Done!')
            continue



        elif 'joke' in inp.lower():
            res = requests.get(
                'https://icanhazdadjoke.com/',
                headers={"Accept":"application/json"}
                )
            if res.status_code == requests.codes.ok:
                print(str(res.json()['joke']))
                continue
            else:
                print('oops!I ran out of jokes')
                continue   
        elif 'email' in inp.lower():
            talkToMe('Who is the recipient?')
            recipient = input('email ID : ')
            talkToMe('What should I say?')
            content = input('Content of the email : ')

            #init gmail SMTP
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('sambitmallick123@gmail.com', 'Konveect1990!')
            #mail.sendmail('Sambit Mallick', 'sambitmallick123@gmail.com', content)
            mail.sendmail('JARVIS Mail',recipient, content)
            mail.close()
            talkToMe('Email sent.')
            continue
        elif 'time' in inp.lower():
            print ("Current date and time : ")
            print (now.strftime("%Y-%m-%d %H:%M:%S"))
            continue
        else:
            results = model.predict([bag_of_words(inp, words)])



        #results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                #print(tag)

        print(random.choice(responses))

chat()