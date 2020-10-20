use regex::Regex;
use std::cmp::min;


pub struct Tokenizer {
    re: Regex,
    lowercase: bool,
    ngram_min: usize,
    ngram_max: usize,
}

impl Tokenizer {
    pub fn new(regex_string: &str, lowercase: bool, ngram_range: (usize, usize)) -> Tokenizer {
        let re = Regex::new(regex_string).unwrap();
        Tokenizer {
            re,
            lowercase,
            ngram_min: ngram_range.0,
            ngram_max: ngram_range.1,
        }
    }

    pub fn tokenize(&self, doc: String) -> Vec<String> {
        // Convert to lowercase if needed
        let doc: String = if self.lowercase {
            doc.to_lowercase()
        } else {
            doc
        };

        let words: Vec<&str> = self.re
            .find_iter(&doc)
            .map(|x| x.as_str())
            .collect();

        let word_count = words.len();

        let ngram_min = self.ngram_min;
        let ngram_max = min(self.ngram_max, word_count);

        let mut tokens: Vec<String> = Vec::with_capacity(
            word_count * (ngram_max - ngram_min + 1)
        );

        for n in ngram_min..ngram_max + 1 {
            for i in 0..word_count - n + 1 {
                tokens.push(
                    words[i..i+n].join(" ")
                );
            }
        }

        tokens
    }
}
