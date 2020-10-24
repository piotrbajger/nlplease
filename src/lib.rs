use std::collections::HashSet;
use pyo3::prelude::*;
use pyo3::types::*;

mod tokenizer;
mod extract_features;
use extract_features::{ Vocabulary, CsrMatrix, TokenBounds };


#[pymodule]
fn nlplease(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "process_corpus")]
    #[allow(clippy::too_many_arguments)]
    fn process_corpus(
        _py: Python,
        py_raw_documents: &PyList,
        lowercase: bool,
        ngram_range: (usize, usize),
        min_doc_freq: Option<f32>,
        max_doc_freq: Option<f32>,
        min_doc_count: Option<i32>,
        max_doc_count: Option<i32>,
        max_features: Option<usize>,
        vocabulary: Vocabulary,
        stopwords: HashSet<String>,
    ) -> (Vocabulary, CsrMatrix) {
        let raw_documents: Vec<String> = py_raw_documents.extract().unwrap();
        let token_bounds = TokenBounds {
            min_doc_freq, max_doc_freq, min_doc_count, max_doc_count
        };
        extract_features::extract_features(
            raw_documents,
            lowercase,
            ngram_range,
            token_bounds,
            max_features,
            vocabulary,
            stopwords,
        )
    }

    Ok(())
}
