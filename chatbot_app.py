# imports
import nltk                                     # used to tokenize and lemmatize sentences
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model             # used as the deep learning model to handle bag of words
import json
import random
import tkinter                                  # used to build user interface
from tkinter import *

# load the trained model and pickle files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')


# function that tokenizes, lemmatizes, and lowercases an input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# function that returns a bag of words for the input sentence 
def bow(sentence, words, show_details=True):
    
    # tokenize the pattern/sentence
    sentence_words = clean_up_sentence(sentence)
    
    # create bag of words
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    
    return(np.array(bag))


# function that returns the probability of the classes given an input sentence and model
def predict_class(sentence, model):
    
    # gets bag of words and predicts classes
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
   
    # filters out class predictions that fall beneath a certain threshold
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
   
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"class": classes[r[0]], "probability": str(r[1])})
    
    return return_list


# function that returns a random response from the json file based on what the predicted class is 
def getResponse(predicted_classes, intents_json):
    tag = predicted_classes[0]['class']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# function that returns a chatbot response given an input sentence
def chatbot_response(text):
    predicted_classes = predict_class(text, model)
    res = getResponse(predicted_classes, intents)
    return res

# function that acts as a send button for the user interface
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# User interface
base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# create chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

# bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# create button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff', command=send)

# create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

# place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

# continuously run the chat window / user interface
base.mainloop()