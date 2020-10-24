import numbers

from sklearn.feature_extraction.text import CountVectorizer
import scipy

import nlplease


class NlpleaseCountVectorizer(CountVectorizer):
    """
    Monkey-patched CountVectorizer which uses Rust
    to do the heavy-lifting.
    """

    def _validate_params(self):
        """
        Validate parameters.

        Prepares the document frequency clipping boundaries
        and sets up stopwords.
        """
        super()._validate_params()

        if isinstance(self.min_df, numbers.Integral):
            self._min_doc_count = self.min_df
            self._min_doc_freq = None
        else:
            self._min_doc_count = None
            self._min_doc_freq = self.min_df
        if isinstance(self.max_df, numbers.Integral):
            self._max_doc_count = self.max_df
            self._max_doc_freq = None
        else:
            self._max_doc_count = None
            self._max_doc_freq = self.max_df

        if self.stop_words is None:
            self.stop_words = set()

    def fit_transform(self, raw_documents, y=None):
        """ "
        Learn the vocabulary dictionary and return the document-term matrix.

        Delegates most of the heavy lifting (tokenization, etc.) to Rust.

        :param list[str] raw_documents: An iterable which yields strings.

        :returns scipy.csr_matrix: A document-term frequency matrix.
        """
        if not isinstance(raw_documents, list):
            raise ValueError(
                "A list of raw text documents expected, " f"got {type(raw_documents)}."
            )

        self._validate_params()
        self._validate_vocabulary()

        if not self.fixed_vocabulary_:
            self.vocabulary_ = {}

        vocabulary, mat = nlplease.process_corpus(
            raw_documents,
            self.lowercase,
            self.ngram_range,
            self._min_doc_freq,
            self._max_doc_freq,
            self._min_doc_count,
            self._max_doc_count,
            self.max_features,
            self.vocabulary_,
            self.stop_words,
        )

        x = scipy.sparse.csr_matrix(mat, shape=(len(mat[2]) - 1, len(vocabulary)))
        x.sort_indices()

        if not self.fixed_vocabulary_:
            self.vocabulary_ = vocabulary

        return x
