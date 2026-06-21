//! Entity extraction + fuzzy matching — port of `entities.py`.
//!
//! `levenshtein_distance` (L69-L97) and `similarity` (L100-L134) are pure and ported with tests.
//! `extract_entities_regex` (L137-L205) is a scaffold returning an empty set until the regex
//! patterns + stopword filters are ported.

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

/// Regex entity extraction (`entities.py` `extract_entities_regex` L137-L205). Scaffold: empty.
pub fn extract_entities_regex(_text: &str) -> Vec<String> {
    // TODO: @handles, #tags, quoted spans, capitalized phrases; stopword/number/dedup filters.
    Vec::new()
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
}
