# Psychiatrist ChatBot

When first running the program, it may ask you to download certain NTLK packages. The only way I found I could download them was by running the code below:

import nltk

import ssl

try:

    _create_unverified_https_context = ssl._create_unverified_context
    
except AttributeError:

    pass
    
else:

    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

Link to source can be found here: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
