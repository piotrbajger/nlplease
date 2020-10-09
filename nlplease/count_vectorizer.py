from sklearn.feature_extraction.text import CountVectorizer
import scipy

import nlplease


class NlpleaseCountVectorizer(CountVectorizer):
    """
    Monkey-patched CountVectorizer which uses Rust
    to do the heavy-lifting.
    """

    def _count_vocab(self, raw_documents, fixed_vocab):
        """
        A patch for the actual tokenization.
        """
        vocabulary, mat = nlplease.count_vocab(
            raw_documents, self.lowercase, self.ngram_range
        )

        x = scipy.sparse.csr_matrix(mat)
        x.sort_indices()

        return vocabulary, x
