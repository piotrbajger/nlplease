from unittest import TestCase

from nlplease import count_vocab

from scipy.sparse import csr_matrix
import numpy as np


class TestCountVocab(TestCase):
    @classmethod
    def setup_class(cls):
        cls.raw_documents = [
            "I like trains",
            "i LIKE trains trains are cool",
            "i do not like trains",
        ]

    def test_count_vocab(self):
        words, mat = count_vocab(
            self.raw_documents,
            lower=True,
            ngram_range=(1, 1)
        )

        result = csr_matrix(mat)
        expected_result = csr_matrix(
            np.array(
                [
                    [1, 1, 0, 0, 0, 0],
                    [1, 2, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1, 1],
                ]
            )
        )
        diff = result != expected_result

        for word in ["like", "trains", "are", "cool", "do", "not"]:
            self.assertIn(word, words)

        self.assertFalse(diff.todense().any())

    def test_count_vocab_ngrams(self):
        words, mat = count_vocab(
            self.raw_documents,
            lower=True,
            ngram_range=(2, 4)
        )

        result = csr_matrix(mat)
        expected_result = csr_matrix(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                ]
            )
        )
        diff = result != expected_result

        expected_words = [
            "like trains",
            "like trains trains",
            "like trains trains are",
            "trains trains",
            "trains trains are",
            "trains trains are cool",
            "trains are",
            "trains are cool",
            "are cool",
            "do not",
            "do not like",
            "do not like trains",
            "not like",
            "not like trains",
        ]
        for word in expected_words:
            self.assertIn(word, expected_words)

        self.assertFalse(diff.todense().any())
