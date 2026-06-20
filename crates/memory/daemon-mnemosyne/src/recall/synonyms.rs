//! Query normalization + synonym expansion — port of `synonyms.py`.
//!
//! `normalize_query` drops stop words, maps to canonical, sorts+dedups (`synonyms.py` L90-L113);
//! `expand_query` builds `(canonical|syn1|syn2)` groups (L116-L143). Scaffold: stop-word + sort
//! normalization is implemented; the 40 `SYNONYM_GROUPS` are a TODO (currently identity mapping).

/// A small stop-word set (subset of `synonyms.py` `STOP_WORDS` L59-L74).
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in", "on", "for",
    "with", "what", "who", "do", "i", "my", "me",
];

/// Normalize a query: lowercase, drop stop words, sort+dedup tokens (`synonyms.py` L90-L113).
pub fn normalize_query(query: &str) -> String {
    let mut words: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|w| !STOP_WORDS.contains(w))
        .map(|w| w.to_string()) // TODO: map via _WORD_TO_CANONICAL
        .collect();
    words.sort();
    words.dedup();
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_stopwords_and_sorts() {
        assert_eq!(normalize_query("What is the auth flow"), "auth flow");
    }
}
