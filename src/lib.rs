use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::wrap_pyfunction;

mod count_vectorizer;


#[pymodule]
fn nlplease(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(count_vocab)).unwrap();

    #[pyfn(m, "count_vocab")]
    fn count_vocab(_py: Python, py_raw_documents: &PyList) -> (Vec<i64>, Vec<i64>, Vec<usize>) {
        let raw_documents: Vec<&str> = py_raw_documents.extract().unwrap();
        count_vectorizer::count_vocab(raw_documents)
    }

    Ok(())
}
