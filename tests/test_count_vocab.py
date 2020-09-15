from unittest import TestCase

from nlplease import count_vocab

from scipy.sparse import csr_matrix
import numpy as np


class TestCountVectorizer(TestCase):

    def test_count_vocab(self):
        raw_documents = [
            "i like trains",
            "i like trains trains are cool",
            "i do not like trains",
        ]

        result = csr_matrix(count_vocab(raw_documents))
        expected_result = csr_matrix(
            np.array([
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 2, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 1],
            ])
        )
        diff = result != expected_result

        self.assertFalse(diff.todense().any())
