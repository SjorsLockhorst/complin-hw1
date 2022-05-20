import os
import unittest

from langdetect import (
    prepare,
    ngrams,
    ngram_table,
    write_ngrams,
    read_ngrams,
    cosine_similarity
)

from match_language import LangMatcher


class TestLangdetect(unittest.TestCase):
    def test_prepare(self):
        test = 'This is <cough,cough> "HAL-9000".  Don\'t touch!'
        tested_tokens = prepare(test)
        expected = ['This', 'is', 'cough',
                    'cough', 'HAL-9000', "Don't", 'touch']
        self.assertEqual(tested_tokens, expected)

    def test_ngrams(self):
        trigrams = ngrams("R2.D2")
        expected = ['R2.', '2.D', '.D2']

        self.assertEqual(trigrams, expected)

    def test_ngrams_table(self):
        result = ngram_table("hiep, hiep, hoera!", n=3, limit=4)
        expected = {'<hi': 2, 'hie': 2, 'ep>': 2, 'iep': 2}
        self.assertEqual(result, expected)

    def test_write_ngrams(self):
        temp_filename = "rover.10.TEMP"
        text = "Het valt voor, dat bij één roveroverval, één rover voorover over één roverval valt."
        table = ngram_table(text, 3, limit=10)
        write_ngrams(table, temp_filename)
        reread_table = read_ngrams(temp_filename)
        os.remove(temp_filename)
        self.assertEqual(table, reread_table)

    def test_cosine_similarity(self):
        table1 = {"<he": 2, "het": 1}
        table2 = {"<he": 2, "hem": 1}
        score = cosine_similarity(table1, table2)
        # The score *should be* 0.8, but we get floating point error
        self.assertLess(abs(score - 4/5), 0.000001)


class TestMatchLanguage(unittest.TestCase):

    def test_language_match(self):
        langmatcher = LangMatcher("./models/2-200")
        self.assertEqual(langmatcher.recognize(
            "./datafiles/training/Ewe-UTF8")[0], "Ewe")
