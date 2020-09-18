from unittest import TestCase

from nlplease import count_vocab

from scipy.sparse import csr_matrix
import numpy as np


class TestCountVocab(TestCase):

    def test_count_vocab(self):
        raw_documents = [
            "I like trains",
            "i LIKE trains trains are cool",
            "i do not like trains",
        ]

        words, mat = count_vocab(raw_documents, lower=True)

        result = csr_matrix(mat)
        expected_result = csr_matrix(
            np.array([
                [1, 1, 0, 0, 0, 0],
                [1, 2, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1],
            ])
        )
        diff = result != expected_result

        for word in ["like", "trains", "are", "cool", "do", "not"]:
            self.assertIn(word, words)

        self.assertFalse(diff.todense().any())
