// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Entity extraction + fuzzy matching — port of `entities.py`.
//!
//! `levenshtein_distance` (L69-L97) and `similarity` (L100-L134) are pure and ported with tests.
//! `extract_entities_regex` (L137-L205) extracts @handles, #tags, quoted spans, and capitalized
//! phrases, then applies the stopword/number/lowercase/substring filters.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Stop words filtered from entity extraction (`entities.py` `ENTITY_EXTRACTION_STOP_WORDS`
/// L19-L39): standard function words plus meta/system noise mined from LLM summaries.
const STOP_WORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "here",
    "there",
    "where",
    "when",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "how",
    "why",
    "assistant",
    "user",
    "skill",
    "review",
    "target",
    "class",
    "level",
    "signals",
    "phase",
    "api",
    "pi",
    "summary",
    "added",
    "active",
    "not",
    "whether",
    "all",
    "no",
    "replying",
    "ai",
    "memory",
    "conversation",
    "fact",
    "false",
    "true",
    "none",
    "null",
    "signal",
    "hermes",
    "agent",
    "model",
    "system",
    "note",
    "task",
    "project",
    "result",
    "output",
    "input",
    "data",
    "step",
    "process",
    "point",
    "way",
    "thing",
    "time",
    "work",
];

fn stop_words() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| STOP_WORDS.iter().copied().collect())
}

/// True if `word` (expected already lowercased) is a meta/system stop word. Shared with the
/// annotation `mentions` noise filter.
pub fn is_stop_word(word: &str) -> bool {
    stop_words().contains(word)
}

/// The ordered entity-extraction patterns (`entities.py` `_ENTITY_PATTERNS` L49-L62). Each capture
/// group 1 is the candidate entity span.
fn entity_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"@(\w{2,30})").unwrap(),
            Regex::new(r"#(\w{2,30})").unwrap(),
            Regex::new(r#""([^"]{2,50})""#).unwrap(),
            Regex::new(r"'([^']{2,50})'").unwrap(),
            Regex::new(r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,4})\b").unwrap(),
            Regex::new(r"\b([A-Z][a-zA-Z]{1,20})\b").unwrap(),
        ]
    })
}

/// Levenshtein edit distance (`entities.py` L69-L97).
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Fuzzy similarity in `[0, 1]` (`entities.py` `similarity` L100-L134): exact 1.0, prefix
/// `0.7 + ratio*0.3`, substring `0.5 + ratio*0.3`, else `1 - dist/max_len`.
pub fn similarity(s1: &str, s2: &str) -> f64 {
    let a = s1.to_lowercase();
    let b = s2.to_lowercase();
    if a == b {
        return 1.0;
    }
    let (shorter, longer) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    let ratio = shorter.len() as f64 / longer.len() as f64;
    if longer.starts_with(shorter.as_str()) && ratio >= 0.3 {
        return 0.7 + ratio * 0.3;
    }
    if longer.contains(shorter.as_str()) {
        return 0.5 + ratio * 0.3;
    }
    let dist = levenshtein_distance(&a, &b);
    let max_len = a.chars().count().max(b.chars().count()).max(1);
    1.0 - (dist as f64) / (max_len as f64)
}

/// Find known entities similar to `entity` above `threshold` (`entities.py` L208-L236).
pub fn find_similar_entities(entity: &str, known: &[String], threshold: f64) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = known
        .iter()
        .map(|k| (k.clone(), similarity(entity, k)))
        .filter(|(_, s)| *s >= threshold)
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// True if every character (ignoring `.` and `,`) is an ASCII digit (`entities.py` L164).
fn is_pure_number(entity: &str) -> bool {
    let stripped: String = entity.chars().filter(|c| *c != '.' && *c != ',').collect();
    !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit())
}

/// Regex entity extraction (`entities.py` `extract_entities_regex` L137-L205). Returns the unique,
/// sorted entity candidates after stopword/number/lowercase filtering and substring dedup.
pub fn extract_entities_regex(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let stop = stop_words();
    let mut entities: HashSet<String> = HashSet::new();

    for pattern in entity_patterns() {
        for caps in pattern.captures_iter(text) {
            let Some(m) = caps.get(1) else { continue };
            let entity = m.as_str().trim();
            if entity.len() < 2 {
                continue;
            }
            let words: Vec<&str> = entity.split_whitespace().collect();
            // Single-word stopword, or any word a stopword (contaminated phrase).
            if words
                .iter()
                .any(|w| stop.contains(w.to_lowercase().as_str()))
            {
                continue;
            }
            if is_pure_number(entity) {
                continue;
            }
            // Standalone lowercase word: keep only when it came from an @mention/#hashtag (the
            // preceding char in the source text is `@` or `#`).
            if words.len() == 1
                && entity.chars().next().is_some_and(|c| c.is_lowercase())
                && !entity.starts_with('@')
                && !entity.starts_with('#')
            {
                let start = m.start();
                let prefix = text[..start].chars().next_back();
                if !matches!(prefix, Some('@') | Some('#')) {
                    continue;
                }
            }
            entities.insert(entity.to_string());
        }
    }

    // Drop entities that are substrings of a longer entity (skipping @mentions/#hashtags).
    let all: Vec<String> = entities.iter().cloned().collect();
    let mut filtered: Vec<String> = all
        .iter()
        .filter(|entity| {
            if entity.starts_with('@') || entity.starts_with('#') {
                return true;
            }
            !all.iter().any(|other| {
                other != *entity
                    && !other.starts_with('@')
                    && !other.starts_with('#')
                    && other.contains(entity.as_str())
            })
        })
        .cloned()
        .collect();
    filtered.sort();
    filtered.dedup();
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn similarity_exact_and_prefix() {
        assert_eq!(similarity("Maya", "maya"), 1.0);
        assert!(similarity("authentication", "auth") > 0.5);
    }

    #[test]
    fn extracts_capitalized_handles_and_quotes() {
        let got = extract_entities_regex("Maya joined Acme. Ping @alice about \"Blue Falcon\".");
        // Capitalized name, org, @mention (lowercased, prefix-allowed), and a quoted multi-word span.
        assert!(got.contains(&"Maya".to_string()), "{got:?}");
        assert!(got.contains(&"Acme".to_string()), "{got:?}");
        assert!(got.contains(&"alice".to_string()), "{got:?}");
        assert!(got.contains(&"Blue Falcon".to_string()), "{got:?}");
    }

    #[test]
    fn filters_stopwords_numbers_and_lowercase() {
        // "The" is a stopword; "42" pure number; bare lowercase "stuff" rejected (no @/# prefix).
        let got = extract_entities_regex("The number is 42 and some stuff happened");
        assert!(
            !got.iter().any(|e| e.eq_ignore_ascii_case("the")),
            "{got:?}"
        );
        assert!(!got.contains(&"42".to_string()), "{got:?}");
        assert!(!got.contains(&"stuff".to_string()), "{got:?}");
    }

    #[test]
    fn drops_substrings_of_longer_entities() {
        let got = extract_entities_regex("New York is in New York State");
        // "New York" is a substring of "New York State" -> only the longer survives.
        assert!(got.contains(&"New York State".to_string()), "{got:?}");
        assert!(!got.contains(&"New York".to_string()), "{got:?}");
    }
}
