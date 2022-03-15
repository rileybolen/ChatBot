import unittest
from nltk.corpus import wordnet
from train import get_pos


class TestPOS(unittest.TestCase):

    def test_adjective1(self):
        word = "happy"
        result = get_pos(word)
        answer = wordnet.ADJ
        self.assertEqual(result, answer)

    def test_adjective2(self):
        word = "short"
        result = get_pos(word)
        answer = wordnet.ADJ
        self.assertEqual(result, answer)

    def test_noun1(self):
        word = "house"
        result = get_pos(word)
        answer = wordnet.NOUN
        self.assertEqual(result, answer)

    def test_noun2(self):
        word = "dad"
        result = get_pos(word)
        answer = wordnet.NOUN
        self.assertEqual(result, answer)

    def test_verb1(self):
        word = "sleeping"
        result = get_pos(word)
        answer = wordnet.VERB
        self.assertEqual(result, answer)

    def test_verb2(self):
        word = "eating"
        result = get_pos(word)
        answer = wordnet.VERB
        self.assertEqual(result, answer)

    def test_adverb1(self):
        word = "accidentally"
        result = get_pos(word)
        answer = wordnet.ADV
        self.assertEqual(result, answer)

    def test_adverb2(self):
        word = "barely"
        result = get_pos(word)
        answer = wordnet.ADV
        self.assertEqual(result, answer)

if __name__ == '__main__':
    unittest.main()