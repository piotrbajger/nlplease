from unittest import TestCase

from nlplease.count_vectorizer import NlpleaseCountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class TestNlpleaseCountVectorizer(TestCase):
    @classmethod
    def setup_class(cls):
        cls.raw_documents = [
            "I like trains",
            "i LIKE trains trains are cool",
            "i do not like trains",
        ]

        # NlpleaseCountVectorizer and CountVectorizer
        # will be run with these parameter sets
        cls.params = [
            dict(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=2),
            dict(lowercase=False, ngram_range=(2, 4), min_df=2, max_df=3),
        ]

    def test_process_corpus_ngrams(self):
        for param_dict in self.params:
            fail_msg = f"Failed for: {param_dict}"

            vect = NlpleaseCountVectorizer(**param_dict)
            ref_vect = CountVectorizer(**param_dict)

            mat = vect.fit_transform(self.raw_documents)
            ref_mat = ref_vect.fit_transform(self.raw_documents)

            self.assertEqual(mat.shape, ref_mat.shape, msg=fail_msg)
            self.assertFalse((mat != ref_mat).todense().any(), msg=fail_msg)
            self.assertEqual(vect.vocabulary_, ref_vect.vocabulary_, msg=fail_msg)
