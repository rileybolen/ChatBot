import spacy
from spacy import displacy
import unittest
from app.chatbot.chatbot_app import ner

class TestNER(unittest.TestCase):
    def test_persons(self):
        self.assertEqual(ner('James is my favourite man!'), 'John is my favourite man !')
        self.assertEqual(ner('the most peaceful man to live was Ghandi'), 'the most peaceful man to live was John')
        self.assertEqual(ner('Emma is my friend'), 'John is my friend')
    
    def test_place(self):
        self.assertEqual(ner('Namibia is a place in Africa'), 'Canada is a place in LOC')
        self.assertEqual(ner('I live near London'), 'I live near Canada')
        self.assertEqual(ner('Have you been to Moscow?'), 'Have you been to Canada ?')
    
    def test_org(self):
        self.assertEqual(ner('do you like Apple?'),'do you like Microsoft ?')
        self.assertEqual(ner('What is the value of Ferrari?'), 'What is the value of Microsoft ?')
        self.assertEqual(ner('Another peep and I will call nato!'), 'Another peep and I will call Microsoft !')
    
    def test_num(self):
        self.assertEqual(ner('I will have 20 more bagels please'), 'I will have three more bagels please')
        self.assertEqual(ner('There should be four more chairs!'), 'There should be three more chairs !')
        self.assertEqual(ner('there are ten scores of soldiers at your disposal'), 'there are three scores of soldiers at your disposal')