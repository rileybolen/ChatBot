import unittest
from nltk.corpus import wordnet
from train import get_synonym

class testsynonym(unittest.TestCase):

    def test_synonym1(self):
        word = "happy"
        result = get_synonym(word)
        answer = ['happy', 'felicitous', 'happy', 'glad', 'happy', 'happy', 'well-chosen']
        self.assertEqual(result, answer)

    def test_synonym2(self):
        word = "sad"
        result = get_synonym(word)
        answer = ['sad', 'sad', 'deplorable', 'distressing', 'lamentable', 'pitiful', 'sad', 'sorry']
        self.assertEqual(result, answer)

    def test_synonym3(self):
        word = "hello"
        result = get_synonym(word)
        answer = ['hello', 'hullo', 'hi', 'howdy', 'how-do-you-do']
        self.assertEqual(result, answer)


if __name__ == '__main__':
    unittest.main()