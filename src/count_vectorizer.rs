use crate::tokenizer::Tokenizer;

use regex::Regex;
use std::collections::HashMap;


pub fn count_vocab(
    raw_documents: Vec<String>, lower: bool, ngram_range: (usize, usize)
) -> (HashMap<String, i64>, (Vec<i64>, Vec<i64>, Vec<usize>)) {
    let re = Regex::new(r"(?u)\b\w\w+\b").unwrap();

    let mut vocabulary: HashMap<String, i64> = HashMap::new();
    let mut word_count: i64 = 0;

    // CSR sparse matrix definition
    let mut data: Vec<i64> = Vec::new();
    let mut indices: Vec<i64> = Vec::new();
    let mut indptr: Vec<usize> = Vec::new();

    indptr.push(0);

    let mut temp_token: &str;

    for doc in raw_documents.iter() {
        let mut counter: HashMap<String, i64> = HashMap::new();
        let doc: String = if lower { doc.to_lowercase() } else { doc.into() };
        let tokenizer = Tokenizer::new(&doc, ngram_range, &re);

        for cap in tokenizer {
            temp_token = cap.as_str();
            // Add word to the vocabulary if not already present
            if !vocabulary.contains_key(temp_token) {
                vocabulary.insert(temp_token.to_string(), word_count);
                word_count += 1;
            }
            // Increase word count in current document
            *counter.entry(temp_token.to_string()).or_insert(0) += 1;
        }

        // Create next CSR matrix row
        for (token, count) in counter.iter() {
            data.push(*count);
            indices.push(*vocabulary.get(token).unwrap());
        }
        indptr.push(data.len());
    }

    (vocabulary, (data, indices, indptr))
}
