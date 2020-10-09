use regex::Regex;
use std::cmp::min;


pub struct Tokenizer<'a> {
    grams: Vec<&'a str>,
    grams_len: usize,
    ngram_max: usize,
    ngram_len: usize,
    ngram_idx: usize,
}


impl<'a> Tokenizer<'a> {
    pub fn new(doc: &'a str, ngram_range: (usize, usize), re: &Regex) -> Tokenizer<'a> {
        let grams: Vec<&str> = re
            .find_iter(doc)
            .map(|x| x.as_str())
            .collect();
        let grams_len = grams.len();

        let ngram_min = ngram_range.0;
        let ngram_max = min(ngram_range.1, grams.len());

        Tokenizer {
            grams,
            grams_len,
            ngram_max,
            ngram_len: ngram_min,
            ngram_idx: 0,
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.ngram_idx + self.ngram_len > self.grams_len {
            self.ngram_idx = 0;
            self.ngram_len += 1;
            if self.ngram_len > self.ngram_max { return None }
        }

        let ngram = self.grams[self.ngram_idx..self.ngram_idx + self.ngram_len].join(" ");

        self.ngram_idx += 1;
        Some(ngram)
    }
}
