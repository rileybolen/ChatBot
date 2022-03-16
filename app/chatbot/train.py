# Imports
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import json
import pickle
from nltk.corpus import wordnet
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

# POS Mapper Function (used to ensure POS compatibility with lemmatizer)
def get_pos(word):
    tag = nltk.pos_tag([word])
    tag = tag[0][1][0].upper()
    dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV }

    return dict.get(tag, wordnet.NOUN)

# Initialize and load the json file
words=[]
classes = []
synonyms = []
documents = []
ignore_words = ['?', '!']
data_file = open('app/chatbot/data/intents.json').read()
intents = json.loads(data_file)

# Add the patterns and their respective classes to the following arrays
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word from the particular pattern and add them to the words array
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # add the tokenized words and it's associated tag to the documents array
        documents.append((w, intent['tag']))

        # add the tag to the classes array (if it's unique)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# function to return a list of synonyms of a given word
def get_synonym(words):
    syns = []
    if len(words) == 0: # if blank is passed, return nothing
        return
    for syn in wordnet.synsets(words.lower()):
        for l in syn.lemmas():
            syns.append(l.name())
        return syns


# lemmatize and lowercase each word, remove duplicates
lemmatizer = WordNetLemmatizer()

# gets the synonyms of a word in words
for word in words:
        # print(word)
    for syn in wordnet.synsets(word.lower()):
        for l in syn.lemmas():
            synonyms.append(l.name())

# add synonyms to words
for k in synonyms:
    words.append(k)

words = [lemmatizer.lemmatize(w.lower(), get_pos(w.lower())) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes (duplicates don't need to be removed because they are already unique)
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = tags
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

# write all the words and unique classes to their own pickle files for later use
pickle.dump(words, open('app/chatbot/data/words.pkl', 'wb'))
pickle.dump(classes, open('app/chatbot/data/classes.pkl', 'wb'))


# initialize the training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# create the training data for each pattern in the documents array
for doc in documents:
    # initialize the bag of words
    bag = []
    # list of tokenized words for the particular pattern
    pattern_words = doc[0]
    # lemmatize each word - in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower(), get_pos(word.lower())) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        if w in pattern_words or (len(pattern_words) > 0 and w in get_synonym(pattern_words[len(pattern_words)-1])):
            bag.append(1) 
        else:
             bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # append the bag of words (input data) and respective class (output data) to the training data array
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - classes
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of classes to predict output class with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('app/chatbot/models/trained_chatbot_model.h5', hist)

print("model created")