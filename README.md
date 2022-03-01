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
