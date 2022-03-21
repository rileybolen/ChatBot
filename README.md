# Psychiatrist ChatBot
## COSC 310 Software Engineering
### Group 21: Pavni Agarwal, Riley Bolen, Gerren Hunter, Graham Itcush, Aidan Murphy and Maxwell Rex

------

### Local Environment Setup

#### Install Dependencies

`pip install nltk`

`pip install numpy`

`pip install keras`

`pip install tensorflow`

`pip install spacy`

download spacy model

`python -m spacy download en_core_web_sm`

When first running the program, it may ask you to download certain NTLK packages. The only way I found I could download them was by running the code below:

```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
```

Link to source can be found here: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed


#### Train ChatBot

To train the chatbot you can run the file `app/chatbot/train.py`

#### Use ChatBot

To use the chatbot you can run the file `app/chatbot/chatbot_app.py`

------

### List of Features (for A3) 

#### Out of topic question handler

The bot has been updated to handle topics that are out of its range, and respond in a way that directs the conversation back to known conversation areas.

<img width="365" src="https://user-images.githubusercontent.com/77344004/159347118-3acbb02b-7fa1-46a6-92b5-55f6d9fd20d5.png">


#### additional conversation topic: Schizophrenia

Several conversation structures which indicate symptoms of schizophrenia have been added to the bots repertoire, in particular hallucinations and general disorganization. There is also referal to proper health resources if the bot detects these symptoms.

<img width="365" src="https://user-images.githubusercontent.com/77344004/159347709-8e5e0503-2af0-4928-8c67-cf9e8affbe94.png">


#### POS Tagger

The POS (Parts of Speech) tagger function generates a tag (noun, verb, adverb, adjective) for every word that gets passed to it. The tag gets passed to the lammetizer function, along with the word, which will then lammetize the word appropriately. Previously, the lammetizer would default every word as a noun. For example, our chatbot will now be able to identify between an "accident" (noun) and "accidentally" (adverb). 

<img width="365" alt="Screen Shot 2022-03-15 at 4 06 19 PM" src="https://user-images.githubusercontent.com/97714788/158486817-fb65ef40-5d77-4530-8a58-9cf0604befb8.png">


#### Synonym Recognition

The Synonym Recognation function uses WordNet (collection of words and vocabulary) to find synonyms of the words used in our json file. These synonyms get added to the words.pk file which are later used to find the similar words of the training data. For example, the chatbot will now understand sad as also pitiful or distressing and answer appropriately.

<img width="365" src="https://user-images.githubusercontent.com/46100533/158677468-2dcbd50c-1b10-4131-ae78-151f4cc01abd.png">


#### Named Entity Recognition

Named Entity Recognition is implemented through [spaCy](https://spacy.io) (it can also be installed using pip for python). It allows the program to differentiate between different named objects, such as Organizations, People, Geo-Political Entities, Quantities, and many more. The chatbot will mainly make use of this by identifying organizations when asking about work habits. The chabot is able to use the context of the sentence and pretrained knowledge to determine that 'K2' and the 'UN' are both organizations, and favour dialogues responding to work:

<img width="365" src="https://user-images.githubusercontent.com/77344004/158700501-0cc9f567-bd46-4411-a240-c04c67ab2a18.png">
