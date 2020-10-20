use crate::tokenizer::Tokenizer;

use std::collections::HashMap;


pub type Vocabulary = HashMap<String, usize>;
pub type CsrMatrix = (Vec<i64>, Vec<usize>, Vec<usize>);

pub struct TokenBounds {
    pub min_doc_freq: Option<f32>,
    pub max_doc_freq: Option<f32>,
    pub min_doc_count: Option<i32>,
    pub max_doc_count: Option<i32>,
}

pub fn extract_features(
    raw_documents: Vec<String>,
    lowercase: bool,
    ngram_range: (usize, usize),
    token_bounds: TokenBounds,
    mut vocabulary: Vocabulary,
) -> (Vocabulary, CsrMatrix) {
    let fixed_vocabulary = !vocabulary.is_empty();

    let mut word_count: usize = 0;

    let tokenizer = Tokenizer::new(
        r"(?u)\b\w\w+\b",
        lowercase,
        ngram_range,
    );

    // CSR sparse matrix definition
    let mut data: Vec<i64> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = Vec::new();

    indptr.push(0);

    let mut temp_token: &str;
    let mut doc_count: usize = 0;

    for doc in raw_documents.iter() {
        doc_count += 1;
        let mut counter: HashMap<usize, i64> = HashMap::new();

        for cap in tokenizer.tokenize(doc.to_string()) {
            temp_token = cap.as_str();
            // Add word to the vocabulary if not already present
            if !vocabulary.contains_key(temp_token) && !fixed_vocabulary {
                vocabulary.insert(temp_token.to_string(), word_count);
                word_count += 1;
            }
            if let Some(token_index ) = vocabulary.get(&cap) {
                *counter.entry(*token_index).or_insert(0) += 1
            };
        }

        // Create next CSR matrix row
        let fixed_counter: Vec<(usize, i64)> = counter.into_iter().collect();
        data.extend(fixed_counter.iter().map(|p| p.1));
        indices.extend(fixed_counter.iter().map(|p| p.0));
        indptr.push(data.len());
    }

    if !fixed_vocabulary {
       limit_features(
           vocabulary,
           &data,
           &indices,
           &indptr,
           doc_count,
           token_bounds,
       )
    }
    else {
       (vocabulary, (data, indices, indptr))
    }
}


fn limit_features(
    vocabulary: Vocabulary,
    data: &[i64],
    indices: &[usize],
    indptr: &[usize],
    doc_count: usize,
    token_bounds: TokenBounds,
) -> (Vocabulary, CsrMatrix) {
    let (min_doc_count, max_doc_count) = calculate_term_count_bounds(doc_count, token_bounds);

    // For each term compute the number of documents it appears in
    // to perform  document frequency filtering
    let mut term_dc: Vec<usize> = vec![0; vocabulary.len()];
    for term_index in indices.iter() {
        term_dc[*term_index] += 1;
    }
    let term_mask: Vec<bool> = term_dc
        .iter()
        .map(|x| min_doc_count <= *x && *x <= max_doc_count)
        .collect();

    let index_mask: Vec<bool> = indices
        .iter()
        .map(|idx| term_mask[*idx])
        .collect();

    // Compute the offsets to move the indices and indptr by
    let mut indptr_offsets: Vec<usize> = index_mask
        .iter()
        .scan(0, |acc, &x| {
            *acc += !x as usize;
            Some(*acc)
        })
        .collect();
    indptr_offsets.insert(0, 0);

    let mut index_offsets: Vec<usize> = term_mask
        .iter()
        .scan(0, |acc, &x| {
            *acc += !x as usize;
            Some(*acc)
        })
        .collect();
    index_offsets.insert(0, 0);

    // Update the vocabulary indices and retain only the used ngrams
    let mut vocabulary_pairs: Vec<(String, usize)> = vocabulary
        .into_iter()
        .filter(|x| term_mask[x.1])
        .map(|x| (x.0, x.1 - index_offsets[x.1]))
        .collect();
    vocabulary_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Create mapping of sorted indices
    let mut index_map = vec![0; vocabulary_pairs.len()];
    for (i, keyval) in vocabulary_pairs.iter().enumerate() {
        index_map[keyval.1] = i;
    }

    // Re-create the vocabulary using the sorted map
    let vocabulary: Vocabulary = vocabulary_pairs
        .into_iter()
        .map(|x| (x.0, index_map[x.1]))
        .collect();

    // Use offsets and index mapping to re-create the CsrMatrix structure
    let data: Vec<i64> = data
        .iter()
        .enumerate()
        .filter(|x| index_mask[x.0])
        .map(|x| *x.1)
        .collect();

    let indices: Vec<usize> = indices
        .iter()
        .enumerate()
        .filter(|x| index_mask[x.0])
        .map(|x| index_map[*x.1 - index_offsets[*x.1]])
        .collect();

    let indptr: Vec<usize> = indptr
        .iter()
        .map(|x| *x - indptr_offsets[*x])
        .collect();

    (vocabulary, (data, indices, indptr))
}


fn calculate_term_count_bounds(
    doc_count: usize,
    token_bounds: TokenBounds,
) -> (usize, usize) {
    let min_doc_count: usize = match token_bounds.min_doc_count {
        Some(x) => x as usize,
        None => match token_bounds.min_doc_freq {
            Some(x) => (x * (doc_count as f32)) as usize,
            None => 1
        }
    };
    let max_doc_count: usize = match token_bounds.max_doc_count {
        Some(x) => x as usize,
        None => match token_bounds.max_doc_freq {
            Some(x) => (x * (doc_count as f32)) as usize,
            None => doc_count
        }
    };

    (min_doc_count, max_doc_count)
}
