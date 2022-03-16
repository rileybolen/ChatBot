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

When first running the program, it may ask you to download certain NLTK packages. The only way I found I could download them was by running the code below:

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

#### POS Tagger

The POS (Parts of Speech) tagger function generates a tag (noun, verb, adverb, adjective) for every word that gets passed to it. The tag gets passed to the lammetizer function, along with the word, which will then lammetize the word appropriately. Previously, the lammetizer would default every word as a noun. For example, our chatbot will now be able to identify between an "accident" (noun) and "accidentally" (adverb). 

<img width="365" alt="Screen Shot 2022-03-15 at 4 06 19 PM" src="https://user-images.githubusercontent.com/97714788/158486817-fb65ef40-5d77-4530-8a58-9cf0604befb8.png">

#### Synonym Recognition

The Synonym Recognation function uses WordNet (collection of words and vocabulary) to find synonyms of the words used in our json file. These synonyms get added to the words.pk file which are later used to find the similar words of the training data. For example, the chatbot will now understand sad as also pitiful or distressing and answer appropriately.

![image](https://user-images.githubusercontent.com/46100533/158677086-1d4d6bcc-546c-4003-8853-03dde645f646.png)




