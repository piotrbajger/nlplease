use pyo3::prelude::*;
use pyo3::types::*;

mod tokenizer;
mod count_vocab;


#[pymodule]
fn nlplease(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "count_vocab")]
    fn count_vocab(
        _py: Python,
        py_raw_documents: &PyList,
        lower: bool,
        ngram_range: (usize, usize)
    ) -> (count_vocab::Vocabulary, count_vocab::CsrMatrix) {
        let raw_documents: Vec<String> = py_raw_documents.extract().unwrap();
        count_vocab::count_vocab(raw_documents, lower, ngram_range)
    }

    Ok(())
}
