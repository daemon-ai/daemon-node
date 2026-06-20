//! MMR diversity rerank — port of `mmr.py`.
//!
//! `mmr = λ*relevance - (1-λ)*max_jaccard` over already-relevance-sorted candidates, λ default 0.7
//! (`mmr.py` L41-L84). Word-level Jaccard (`mmr.py` L24-L38).

use std::collections::HashSet;

/// Default MMR lambda (`mmr.py` L43).
pub const DEFAULT_LAMBDA: f64 = 0.7;

/// Word-level Jaccard similarity (`mmr.py` `_jaccard_similarity` L24-L38). Empty sets -> 0.0.
pub fn jaccard(a: &str, b: &str) -> f64 {
    let wa: HashSet<&str> = a.split_whitespace().collect();
    let wb: HashSet<&str> = b.split_whitespace().collect();
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    let inter = wa.intersection(&wb).count() as f64;
    let union = wa.union(&wb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Greedy MMR rerank over `(content, relevance)` candidates, returning indices in selected order
/// (`mmr.py` `mmr_rerank` L41-L84).
pub fn mmr_rerank(items: &[(String, f64)], lambda: f64, top_k: usize) -> Vec<usize> {
    if items.is_empty() {
        return Vec::new();
    }
    // Start from the highest-relevance item.
    let mut remaining: Vec<usize> = (0..items.len()).collect();
    remaining.sort_by(|&i, &j| {
        items[j]
            .1
            .partial_cmp(&items[i].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut selected: Vec<usize> = Vec::new();
    while !remaining.is_empty() && selected.len() < top_k {
        let mut best_pos = 0usize;
        let mut best_mmr = f64::NEG_INFINITY;
        for (pos, &cand) in remaining.iter().enumerate() {
            let relevance = items[cand].1;
            let max_sim = selected
                .iter()
                .map(|&s| jaccard(&items[cand].0, &items[s].0))
                .fold(0.0_f64, f64::max);
            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_pos = pos;
            }
        }
        selected.push(remaining.remove(best_pos));
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_overlap() {
        assert!((jaccard("the cat sat", "the cat ran") - 0.5).abs() < 1e-9);
        assert_eq!(jaccard("", "x"), 0.0);
    }

    #[test]
    fn rerank_prefers_diverse() {
        let items = vec![
            ("alpha beta".to_string(), 1.0),
            ("alpha beta".to_string(), 0.95), // near-duplicate of #0
            ("gamma delta".to_string(), 0.9), // diverse
        ];
        let order = mmr_rerank(&items, DEFAULT_LAMBDA, 2);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 2); // diverse item beats the near-duplicate
    }
}
