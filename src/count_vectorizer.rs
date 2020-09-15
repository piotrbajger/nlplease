use regex::Regex;
use std::collections::HashMap;


pub fn count_vocab(raw_documents: Vec<&str>) -> (Vec<i64>, Vec<i64>, Vec<usize>) {
    let re = Regex::new(r"^?\b\w+\b").unwrap();

    let mut vocabulary: HashMap<String, i64> = HashMap::new();
    let mut word_count: i64 = 0;

    // CSR sparse matrix definition
    let mut data: Vec<i64> = Vec::new();
    let mut indices: Vec<i64> = Vec::new();
    let mut indptr: Vec<usize> = Vec::new();

    indptr.push(0);

    for doc in raw_documents.iter() {
        let mut counter: HashMap<String, i64> = HashMap::new();

        for token in re.captures_iter(doc) {
            // Add word to the vocabulary if not already present
            if !vocabulary.contains_key(&token[0]) {
                vocabulary.insert(token[0].to_string(), word_count);
                word_count += 1;
            }
            // Increase word count in current document
            *counter.entry(token[0].to_string()).or_insert(0) += 1;
        }

        for (token, count) in counter.iter() {
            data.push(*count);
            indices.push(*vocabulary.get(token).unwrap());
        }
        indptr.push(data.len());
    }

    (data, indices, indptr)
}
