// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Polyphonic recall — port of `polyphonic_recall.py` (`MNEMOSYNE_POLYPHONIC_RECALL=1`).
//!
//! Four voices (vector/graph/fact/temporal) fused by Reciprocal Rank Fusion. The voices are
//! gathered in [`crate::engine::Engine::recall_polyphonic`] (they need DB access); this module owns
//! the pure fusion (`_combine_voices` L694-L735), diversity rerank (`_diversity_rerank`
//! L737-L767), and budgeted context assembly (`_assemble_context` L783-L803).

use std::collections::HashMap;

/// RRF constant (`polyphonic_recall.py` L701).
pub const RRF_K: f64 = 60.0;

/// The rank assigned to ids missing from a voice's rank map (`polyphonic_recall.py` L729).
pub const MISSING_RANK: usize = 999;

/// RRF contribution for a 1-based rank: `1 / (RRF_K + rank)` (`polyphonic_recall.py` L730).
pub fn rrf_contribution(rank: usize) -> f64 {
    1.0 / (RRF_K + rank as f64)
}

/// Documented voice weights (`polyphonic_recall.py` L128-L133). NOTE: in Python these are
/// **metadata only** — fusion is pure RRF. Kept for stats/parity.
pub const VOICE_WEIGHTS: &[(&str, f64)] = &[
    ("vector", 0.35),
    ("graph", 0.25),
    ("fact", 0.25),
    ("temporal", 0.15),
];

/// A single voice hit (`polyphonic_recall.py` `RecallResult`): a memory id, the voice-local score
/// (used only for intra-voice ranking), the per-row voice label — the graph voice emits both
/// `graph` and `graph_traversal` rows — and a small metadata bag (merged across voices and counted
/// against the context budget).
#[derive(Clone, Debug)]
pub struct VoiceHit {
    /// The memory id this voice surfaced (or a synthetic `cf_...` fact id).
    pub memory_id: String,
    /// The voice-local score (ranks within the voice; absolute value is irrelevant post-RRF).
    pub score: f64,
    /// The per-row voice label.
    pub voice: &'static str,
    /// Voice-specific metadata (`RecallResult.metadata`).
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// A fused candidate (`polyphonic_recall.py` `PolyphonicResult`).
#[derive(Clone, Debug)]
pub struct Fused {
    /// The memory id.
    pub memory_id: String,
    /// The combined RRF score across voices.
    pub combined_score: f64,
    /// Per-voice-label RRF contributions (`voice_scores`; last write wins per label).
    pub voice_scores: HashMap<String, f64>,
    /// Merged metadata across every contributing hit (`metadata.update`, last write per key).
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Combine per-voice hit lists with Reciprocal Rank Fusion (`polyphonic_recall.py`
/// `_combine_voices` L694-L735), preserving the Python quirks exactly:
///
/// - Each *list*'s rank map is keyed by the voice label of its top-scored row, but rank lookups
///   use the label of the list's *first appended* row. For the graph list (which mixes `graph`
///   and `graph_traversal` rows) these can differ: when a traversal row outscores every direct
///   graph row the rank map is keyed `graph_traversal` while lookups ask for `graph`, so every
///   row in that list falls back to [`MISSING_RANK`].
/// - Duplicate ids within one list contribute once per occurrence (no dedup), all at the same
///   (last-written) rank.
/// - `voice_scores` is keyed by the per-row label and last-write-wins; metadata merges per key.
pub fn combine_voices(voice_lists: &[Vec<VoiceHit>]) -> HashMap<String, Fused> {
    // Step 1: rank each list's hits by score descending (stable, like Python's sorted()); key the
    // map by the top row's voice label; later duplicates overwrite earlier ranks.
    let mut voice_ranks: HashMap<&'static str, HashMap<String, usize>> = HashMap::new();
    for hits in voice_lists {
        if hits.is_empty() {
            continue;
        }
        let mut sorted: Vec<&VoiceHit> = hits.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let key = sorted[0].voice;
        let ranks = voice_ranks.entry(key).or_default();
        for (idx, hit) in sorted.iter().enumerate() {
            ranks.insert(hit.memory_id.clone(), idx + 1);
        }
    }

    // Step 2: accumulate RRF contributions, looking ranks up under the label of each list's
    // first row in append order (`polyphonic_recall.py` L716-L733).
    let mut combined: HashMap<String, Fused> = HashMap::new();
    for hits in voice_lists {
        let Some(first) = hits.first() else { continue };
        let lookup = voice_ranks.get(first.voice);
        for hit in hits {
            let rank = lookup
                .and_then(|m| m.get(&hit.memory_id).copied())
                .unwrap_or(MISSING_RANK);
            let contribution = rrf_contribution(rank);
            let entry = combined
                .entry(hit.memory_id.clone())
                .or_insert_with(|| Fused {
                    memory_id: hit.memory_id.clone(),
                    combined_score: 0.0,
                    voice_scores: HashMap::new(),
                    metadata: serde_json::Map::new(),
                });
            entry
                .voice_scores
                .insert(hit.voice.to_string(), contribution);
            entry.combined_score += contribution;
            for (k, v) in &hit.metadata {
                entry.metadata.insert(k.clone(), v.clone());
            }
        }
    }
    combined
}

/// Diversity rerank (`polyphonic_recall.py` `_diversity_rerank` L737-L767): sort by combined
/// score, then greedily keep candidates whose voice-set Jaccard against every already-selected
/// candidate stays at or below `0.8`.
pub fn diversity_rerank(fused: HashMap<String, Fused>, top_k: usize) -> Vec<Fused> {
    let mut sorted: Vec<Fused> = fused.into_values().collect();
    sorted.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut selected: Vec<Fused> = Vec::new();
    for cand in sorted {
        if selected.len() >= top_k {
            break;
        }
        let diverse = selected
            .iter()
            .all(|sel| voice_jaccard(&cand.voice_scores, &sel.voice_scores) <= 0.8);
        if diverse {
            selected.push(cand);
        }
    }
    selected
}

/// Budgeted context assembly (`polyphonic_recall.py` `_assemble_context` L783-L803): admit
/// results until the running character estimate (`len(str(metadata)) + 100` per row — here the
/// serialized metadata, an equivalent-scale stand-in for Python's dict repr) exceeds `budget * 4`
/// chars (~4 chars/token).
pub fn assemble_context(results: Vec<Fused>, budget: usize) -> Vec<Fused> {
    let mut current_chars = 0usize;
    let mut selected = Vec::new();
    for result in results {
        let result_chars = serde_json::Value::Object(result.metadata.clone())
            .to_string()
            .len()
            + 100;
        if current_chars + result_chars > budget * 4 {
            break;
        }
        selected.push(result);
        current_chars += result_chars;
    }
    selected
}

/// Jaccard similarity of two voice-label sets (`_estimate_similarity` L769-L781).
fn voice_jaccard(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<&String> = a.keys().collect();
    let sb: std::collections::HashSet<&String> = b.keys().collect();
    sa.intersection(&sb).count() as f64 / sa.union(&sb).count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f64, voice: &'static str) -> VoiceHit {
        VoiceHit {
            memory_id: id.to_string(),
            score,
            voice,
            metadata: serde_json::Map::new(),
        }
    }

    #[test]
    fn rrf_rewards_cross_voice_agreement() {
        // "a" appears in two voices (rank 1 each); "b" only in one. "a" must win.
        let voices = vec![
            vec![hit("a", 0.9, "vector"), hit("b", 0.8, "vector")],
            vec![hit("a", 0.5, "fact")],
        ];
        let fused = combine_voices(&voices);
        let a = &fused["a"];
        let b = &fused["b"];
        assert!(a.combined_score > b.combined_score);
        assert_eq!(a.voice_scores.len(), 2);
        assert!((a.combined_score - (rrf_contribution(1) * 2.0)).abs() < 1e-12);
        assert!((b.combined_score - rrf_contribution(2)).abs() < 1e-12);
    }

    #[test]
    fn graph_traversal_outranking_graph_hits_falls_back_to_missing_rank() {
        // Python quirk: the rank map keys by the top-sorted row's voice, the lookup by the first
        // appended row's voice. A traversal row outscoring all direct graph rows (0.4 > 0.3)
        // keys the map "graph_traversal" while the lookup asks for "graph" -> every row in the
        // list scores at MISSING_RANK.
        let graph_list = vec![hit("f1", 0.3, "graph"), hit("t1", 0.4, "graph_traversal")];
        let fused = combine_voices(&[graph_list]);
        for id in ["f1", "t1"] {
            assert!(
                (fused[id].combined_score - rrf_contribution(MISSING_RANK)).abs() < 1e-12,
                "{id} must fall back to the missing rank"
            );
        }
        // With a gist (0.6) on top the map keys "graph" and real ranks apply again.
        let graph_list = vec![
            hit("g1", 0.6, "graph"),
            hit("t1", 0.4, "graph_traversal"),
            hit("f1", 0.3, "graph"),
        ];
        let fused = combine_voices(&[graph_list]);
        assert!((fused["g1"].combined_score - rrf_contribution(1)).abs() < 1e-12);
        assert!((fused["t1"].combined_score - rrf_contribution(2)).abs() < 1e-12);
        assert!((fused["f1"].combined_score - rrf_contribution(3)).abs() < 1e-12);
    }

    #[test]
    fn duplicate_ids_within_a_voice_accumulate() {
        // Two entities surfacing the same gist append it twice; both occurrences contribute at
        // the (single) rank the sort assigned (`polyphonic_recall.py` has no intra-voice dedup).
        let graph_list = vec![hit("g1", 0.6, "graph"), hit("g1", 0.6, "graph")];
        let fused = combine_voices(&[graph_list]);
        // Later duplicate overwrites the rank map entry (rank 2), and both adds use it.
        assert!((fused["g1"].combined_score - rrf_contribution(2) * 2.0).abs() < 1e-12);
    }

    #[test]
    fn diversity_drops_same_voice_set() {
        let mut fused = HashMap::new();
        for (id, score) in [("a", 0.5), ("b", 0.4)] {
            fused.insert(
                id.to_string(),
                Fused {
                    memory_id: id.to_string(),
                    combined_score: score,
                    voice_scores: HashMap::from([("vector".to_string(), score)]),
                    metadata: serde_json::Map::new(),
                },
            );
        }
        // Identical voice sets -> Jaccard 1.0 > 0.8 -> second dropped.
        let out = diversity_rerank(fused, 10);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].memory_id, "a");
    }

    #[test]
    fn context_budget_truncates() {
        let fused: Vec<Fused> = (0..5)
            .map(|i| Fused {
                memory_id: format!("m{i}"),
                combined_score: 1.0 - i as f64 * 0.1,
                voice_scores: HashMap::new(),
                metadata: serde_json::Map::new(),
            })
            .collect();
        // Each empty-metadata row costs len("{}") + 100 = 102 chars; a budget of 51 tokens
        // (204 chars) admits exactly two rows.
        let out = assemble_context(fused, 51);
        assert_eq!(out.len(), 2);
    }
}
