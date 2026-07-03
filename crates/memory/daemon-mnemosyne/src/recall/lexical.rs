// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Recall-side lexical helpers — the tokenizer, relevance scorer, fact matcher, and the CJK /
//! Cyrillic fallbacks (`beam.py` L1453-L1700, L2185-L2420). All pure text functions; the SQL
//! search paths that consume them live in `engine/recall.rs`.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Stopwords excluded from recall tokens and strict fact matching (`beam.py`
/// `_FACT_MATCH_STOPWORDS` L1453-L1466).
const FACT_MATCH_STOPWORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "related",
    "should",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "totally",
    "unrelated",
    "use",
    "uses",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "again",
    "into",
    "not",
    "please",
    "somewhere",
    "supposed",
    "them",
    "then",
    "they",
    "whatever",
];

fn fact_match_stopwords() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| FACT_MATCH_STOPWORDS.iter().copied().collect())
}

/// The Unicode-aware recall token pattern (`beam.py` `_RECALL_TOKEN_RE` L1475): word characters
/// (minus `_`) optionally chained through path/version separators when followed by another
/// alphanumeric ("v1.2", "a/b", "snake_case" stays one token).
fn recall_token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\W_][\w]*(?:[_.:/+-]+[^\W_][\w]*)*").unwrap())
}

/// Meaningful lexical tokens for precision gates and fallback scoring (`beam.py` `_recall_tokens`
/// L1492-L1498): lowercased regex tokens, length >= 3, non-stopword, non-numeric.
pub fn recall_tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    recall_token_re()
        .find_iter(&lower)
        .map(|m| m.as_str().to_string())
        .filter(|t| {
            t.chars().count() >= 3
                && !fact_match_stopwords().contains(t.as_str())
                && !t.chars().all(|c| c.is_ascii_digit())
        })
        .collect()
}

/// Query tokens plus a bounded, order-preserving synonym expansion (`beam.py`
/// `_expanded_query_tokens` L1501-L1514).
pub fn expanded_query_tokens(tokens: &[String]) -> Vec<String> {
    let mut expanded: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for token in tokens {
        if seen.insert(token.clone()) {
            expanded.push(token.clone());
        }
        for syn in super::synonyms::recall_synonyms(token) {
            if seen.insert((*syn).to_string()) {
                expanded.push((*syn).to_string());
            }
        }
    }
    expanded
}

/// Meaningful tokens for strict fact matching (`beam.py` `_fact_match_tokens` L1530-L1532).
pub fn fact_match_tokens(text: &str) -> HashSet<String> {
    recall_tokens(text).into_iter().collect()
}

/// True when `ch` is in the CJK ranges beam checks (Han, kana, Hangul; `beam.py` L1536-L1541).
fn is_cjk_char(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
        || ('\u{3040}'..='\u{30ff}').contains(&ch)
        || ('\u{ac00}'..='\u{d7af}').contains(&ch)
}

/// Whether the text contains any CJK characters (`beam.py` `_has_cjk` L2195-L2203).
pub fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

/// The unique CJK characters of a text, sorted (`beam.py` `_cjk_like_search` char set L2216).
pub fn cjk_chars(text: &str) -> Vec<char> {
    let set: std::collections::BTreeSet<char> = text.chars().filter(|c| is_cjk_char(*c)).collect();
    set.into_iter().collect()
}

/// Whether the text contains Russian/Cyrillic characters (`beam.py` `_has_cyrillic` L2293; the
/// `[а-яёА-ЯЁ]` class, deliberately excluding Serbian/Mongolian extras).
pub fn has_cyrillic(text: &str) -> bool {
    text.chars()
        .any(|c| ('а'..='я').contains(&c) || ('А'..='Я').contains(&c) || c == 'ё' || c == 'Ё')
}

/// The words considered by the Cyrillic fallback (`[а-яёa-z0-9]+`, `beam.py` L2360/L2410).
pub fn cyrillic_words(text: &str, min_len: usize) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[а-яёa-z0-9]+").unwrap());
    let lower = text.to_lowercase();
    re.find_iter(&lower)
        .map(|m| m.as_str().to_string())
        .filter(|w| w.chars().count() >= min_len)
        .collect()
}

/// The set of length-`n` sliding character n-grams; whole string when shorter (`beam.py` `_ngrams`
/// L2313-L2321).
fn ngrams(s: &str, n: usize) -> HashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        let mut set = HashSet::new();
        set.insert(s.to_string());
        return set;
    }
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

/// Trigram-Jaccard score in `[0, 1]` for Russian/Cyrillic recall (`beam.py` `_cyrillic_score`
/// L2324-L2352): per query word, the best-matching content word by n-gram Jaccard, averaged.
pub fn cyrillic_score(query: &str, content: &str) -> f64 {
    let q_words = cyrillic_words(query, 3);
    let c_words = cyrillic_words(content, 3);
    if q_words.is_empty() || c_words.is_empty() {
        return 0.0;
    }
    let c_ngrams: Vec<HashSet<String>> = c_words.iter().map(|w| ngrams(w, 3)).collect();
    let mut total = 0.0;
    for qw in &q_words {
        let q_ng = ngrams(qw, 3);
        let mut best = 0.0f64;
        for c_ng in &c_ngrams {
            let inter = q_ng.intersection(c_ng).count();
            let union = q_ng.union(c_ng).count();
            if union == 0 {
                continue;
            }
            let jacc = inter as f64 / union as f64;
            if jacc > best {
                best = jacc;
            }
        }
        total += best;
    }
    total / q_words.len() as f64
}

/// FTS5-safe quoted terms for a natural-language query (`beam.py` `_fts_query_terms` L2185-L2192):
/// the synonym-expanded recall tokens, each `"`-escaped and double-quoted.
pub fn fts_query_terms(query: &str) -> Vec<String> {
    expanded_query_tokens(&recall_tokens(query))
        .into_iter()
        .filter_map(|t| {
            let t = t.replace('"', "\"\"");
            let t = t.trim();
            if t.is_empty() {
                None
            } else {
                Some(format!("\"{t}\""))
            }
        })
        .collect()
}

/// Conservative lexical score in `[0, 1]` (`beam.py` `_lexical_relevance` L1573-L1638): exact
/// token hits, synonym partials (`+0.75`), `>=4`-char token-substring partials (`+0.4`), a
/// whole-query substring bonus (`+1.0`), normalized by the query token count; zero-score queries
/// fall back to CJK character overlap.
pub fn lexical_relevance(query_tokens: &[String], content: &str, query_lower: &str) -> f64 {
    let content_lower = content.to_lowercase();
    let query_cjk: HashSet<char> = query_lower.chars().filter(|c| is_cjk_char(*c)).collect();
    if query_tokens.is_empty() && query_cjk.is_empty() {
        return 0.0;
    }
    // Content tokens, expanded by splitting separators so snake_case/path keys match
    // natural-language queries (`beam.py` L1592-L1601).
    let mut content_tokens: HashSet<String> = recall_tokens(&content_lower).into_iter().collect();
    static SEP_RE: OnceLock<Regex> = OnceLock::new();
    let sep = SEP_RE.get_or_init(|| Regex::new(r"[_:/.-]+").unwrap());
    let parts: Vec<String> = content_tokens
        .iter()
        .flat_map(|t| sep.split(t).map(String::from).collect::<Vec<_>>())
        .filter(|p| {
            p.chars().count() >= 3
                && !fact_match_stopwords().contains(p.as_str())
                && !p.chars().all(|c| c.is_ascii_digit())
        })
        .collect();
    content_tokens.extend(parts);
    if content_tokens.is_empty() && query_cjk.is_empty() {
        return 0.0;
    }

    let exact = query_tokens
        .iter()
        .filter(|t| content_tokens.contains(t.as_str()))
        .count() as f64;
    let mut partial = 0.0;
    for token in query_tokens {
        if content_tokens.contains(token.as_str()) {
            continue;
        }
        let syns = super::synonyms::recall_synonyms(token);
        if !syns.is_empty() && syns.iter().any(|s| content_tokens.contains(*s)) {
            partial += 0.75;
            continue;
        }
        if token.chars().count() >= 4
            && content_tokens
                .iter()
                .filter(|c| c.chars().count() >= 4)
                .any(|c| c.contains(token.as_str()) || token.contains(c.as_str()))
        {
            partial += 0.4;
        }
    }

    let full_match = if !query_lower.is_empty() && content_lower.contains(query_lower) {
        1.0
    } else {
        0.0
    };
    let mut score = (exact + partial + full_match) / query_tokens.len().max(1) as f64;

    if score == 0.0 && !query_cjk.is_empty() {
        let content_cjk: HashSet<char> =
            content_lower.chars().filter(|c| is_cjk_char(*c)).collect();
        score = query_cjk.intersection(&content_cjk).count() as f64 / query_cjk.len() as f64;
    }

    score.min(1.0)
}

/// Conservative fact matching for natural-language recall queries (`beam.py`
/// `_strict_fact_matches` L1642-L1682): exact phrase, then `>=2` meaningful token overlaps, or a
/// single highly-distinctive token (structured `>=8` chars anywhere; plain `>=5` chars only for
/// `<=2`-token lookup queries).
pub fn strict_fact_matches(query_lower: &str, fact_text_lower: &str) -> bool {
    let query = query_lower.trim();
    let fact = fact_text_lower.trim();
    if query.is_empty() || fact.is_empty() {
        return false;
    }
    if fact.contains(query) {
        return true;
    }
    let query_tokens = fact_match_tokens(query);
    let fact_tokens = fact_match_tokens(fact);
    if query_tokens.is_empty() || fact_tokens.is_empty() {
        return false;
    }
    let overlap: Vec<&String> = query_tokens.intersection(&fact_tokens).collect();
    if overlap.len() >= 2 {
        return true;
    }
    if overlap.len() == 1 {
        let token = overlap[0];
        if token.chars().count() >= 8 && token.chars().any(|c| "./:-_".contains(c)) {
            return true;
        }
        if query_tokens.len() <= 2 {
            return token.chars().count() >= 5;
        }
        return false;
    }
    false
}

/// Round to four decimal places (Python's `round(x, 4)` on recall result fields).
pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Truncate to `n` characters (Python's `content[:n]` slices characters, not bytes).
pub fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_keep_diacritics_and_separator_chains() {
        // The pre-Unicode regex split "Stoßlüften" into fragments; parity keeps it whole.
        assert_eq!(
            recall_tokens("Stoßlüften im Bürgeramt:"),
            vec!["stoßlüften".to_string(), "bürgeramt".to_string()]
        );
        // Separator chains survive when followed by alphanumerics; trailing punctuation drops.
        assert_eq!(
            recall_tokens("use v1.2/beta now!"),
            vec!["v1.2/beta".to_string(), "now".to_string()]
        );
        // Stopwords and pure digits are filtered.
        assert!(recall_tokens("the 12345 of").is_empty());
    }

    #[test]
    fn expanded_tokens_dedup_in_order() {
        let tokens = vec!["branding".to_string(), "brand".to_string()];
        let expanded = expanded_query_tokens(&tokens);
        assert_eq!(expanded[0], "branding");
        assert!(expanded.contains(&"positioning".to_string()));
        // "brand" appears once even though it is both input and synonym.
        assert_eq!(expanded.iter().filter(|t| *t == "brand").count(), 1);
    }

    #[test]
    fn lexical_relevance_matches_python_shapes() {
        let q = recall_tokens("auth flow");
        assert!((lexical_relevance(&q, "the auth flow uses jwt", "auth flow") - 1.0).abs() < 1e-9);
        assert!((lexical_relevance(&q, "the auth subsystem", "auth flow") - 0.5).abs() < 1e-9);
        assert_eq!(
            lexical_relevance(&q, "completely unrelated words", "auth flow"),
            0.0
        );
        assert_eq!(lexical_relevance(&[], "anything", ""), 0.0);
        // Separator-split content tokens give full credit for snake_case keys.
        let q2 = recall_tokens("telemetry latency");
        assert!(lexical_relevance(&q2, "telemetry_api_latency_ms: 250", "telemetry latency") > 0.9);
    }

    #[test]
    fn cjk_fallback_scores_char_overlap() {
        let q = "日本語";
        assert!(has_cjk(q));
        let tokens = recall_tokens(q);
        // Full CJK overlap -> 1.0 even with zero ASCII tokens.
        assert!((lexical_relevance(&tokens, "日本語のテスト", q) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cyrillic_trigram_scoring_matches_inflections() {
        assert!(has_cyrillic("тёмная тема"));
        // Inflected forms of the same lemma: "тёмная/тёмную" share 2 of 6 trigrams (1/3) and
        // "тема/тему" share 1 of 3 (1/3), averaging exactly 1/3 — above the 0.3 admission
        // threshold the Cyrillic LIKE fallback applies (`beam.py` L2374).
        let s = cyrillic_score("тёмная тема", "пользователь предпочитает тёмную тему");
        assert!((s - 1.0 / 3.0).abs() < 1e-9, "expected 1/3, got {s}");
        assert!(s > 0.3, "must clear the fallback admission threshold");
        assert_eq!(cyrillic_score("hello", "world"), 0.0);
    }

    #[test]
    fn strict_fact_matcher_requires_distinctive_overlap() {
        // Two meaningful overlaps -> match.
        assert!(strict_fact_matches(
            "hermes memory",
            "hermes uses memory daily"
        ));
        // One short common word -> no match for broad queries.
        assert!(!strict_fact_matches(
            "where is the public context reply",
            "public announcements policy"
        ));
        // Single distinctive structured token matches anywhere.
        assert!(strict_fact_matches(
            "check github.com/acme repo",
            "repo lives at github.com/acme"
        ));
        // Lookup-style short query accepts a 5+ char token.
        assert!(strict_fact_matches("hermes", "hermes is the assistant"));
    }

    #[test]
    fn fts_terms_are_quoted_and_expanded() {
        let terms = fts_query_terms("database preference");
        assert!(terms.contains(&"\"database\"".to_string()));
        // `_RECALL_SYNONYMS` expansion (preference -> prefer ...).
        assert!(terms.contains(&"\"prefer\"".to_string()));
        // Short CJK runs (< 3 chars) produce no FTS terms -> the LIKE fallback fires.
        assert!(fts_query_terms("日本").is_empty());
    }
}
