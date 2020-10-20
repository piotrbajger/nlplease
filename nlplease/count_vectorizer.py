import numbers

from sklearn.feature_extraction.text import CountVectorizer
import scipy

import nlplease


class NlpleaseCountVectorizer(CountVectorizer):
    """
    Monkey-patched CountVectorizer which uses Rust
    to do the heavy-lifting.
    """

    def fit_transform(self, raw_documents, y=None):
        """ "
        Learn the vocabulary dictionary and return the document-term matrix.

        Delegates most of the heavy lifting (tokenization, etc.) to Rust.

        :param list[str] raw_documents: An iterable which yields strings.

        :returns scipy.csr_matrix: A document-term frequency matrix.
        """
        if not isinstance(raw_documents, list):
            raise ValueError(
                "A list of raw text documents expected, "
                f"got {type(raw_documents)}."
            )

        self._validate_params()
        self._validate_vocabulary()

        if isinstance(self.min_df, numbers.Integral):
            min_doc_count = self.min_df
            min_doc_freq = None
        else:
            min_doc_count = None
            min_doc_freq = self.min_df
        if isinstance(self.max_df, numbers.Integral):
            max_doc_count = self.max_df
            max_doc_freq = None
        else:
            max_doc_count = None
            max_doc_freq = self.max_df

        if not self.fixed_vocabulary_:
            self.vocabulary_ = {}

        vocabulary, mat = nlplease.process_corpus(
            raw_documents,
            self.lowercase,
            self.ngram_range,
            min_doc_freq,
            max_doc_freq,
            min_doc_count,
            max_doc_count,
            self.vocabulary_,
        )

        x = scipy.sparse.csr_matrix(mat, shape=(len(mat[2]) - 1, len(vocabulary)))
        x.sort_indices()

        if not self.fixed_vocabulary_:
            self.vocabulary_ = vocabulary

        return x
