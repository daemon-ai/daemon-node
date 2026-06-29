// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Polyphonic recall — port of `polyphonic_recall.py` (P2, `MNEMOSYNE_POLYPHONIC_RECALL=1`).
//!
//! Four voices (vector/graph/fact/temporal) fused by Reciprocal Rank Fusion. The voices are
//! gathered in [`crate::engine::Engine::recall_polyphonic`] (they need DB access); this module owns
//! the pure RRF fusion (`_combine_voices` L694-L735) and diversity rerank (`_diversity_rerank`
//! L737-L781).

use std::collections::HashMap;

/// RRF constant (`polyphonic_recall.py` L701).
pub const RRF_K: f64 = 60.0;

/// RRF contribution for a 1-based rank: `1 / (RRF_K + rank)` (`polyphonic_recall.py` L712-L729).
pub fn rrf_contribution(rank: usize) -> f64 {
    1.0 / (RRF_K + rank as f64)
}

/// Documented voice weights (`polyphonic_recall.py` L128-L133). NOTE: in Python these are
/// **metadata only** — fusion is pure RRF. Kept here for stats/parity.
pub const VOICE_WEIGHTS: &[(&str, f64)] = &[
    ("vector", 0.35),
    ("graph", 0.25),
    ("fact", 0.25),
    ("temporal", 0.15),
];

/// A single voice's hit: a memory id and the voice-local score used only for intra-voice ranking.
#[derive(Clone, Debug)]
pub struct VoiceHit {
    /// The memory id this voice surfaced.
    pub memory_id: String,
    /// The voice-local score (ranks within the voice; absolute value is irrelevant post-RRF).
    pub score: f64,
}

/// A fused candidate: a memory id, its summed RRF score, and the set of voices that surfaced it.
#[derive(Clone, Debug)]
pub struct Fused {
    /// The memory id.
    pub memory_id: String,
    /// The combined RRF score across voices.
    pub combined_score: f64,
    /// Which voices contributed (used by the diversity rerank).
    pub voices: Vec<String>,
}

/// Combine per-voice hits with Reciprocal Rank Fusion (`polyphonic_recall.py` `_combine_voices`
/// L694-L735): rank each voice's hits by score, add `1/(RRF_K + rank)` per appearance, and track
/// which voices touched each id. Returned sorted by combined score descending.
pub fn fuse(voices: &[(&str, Vec<VoiceHit>)]) -> Vec<Fused> {
    let mut combined: HashMap<String, Fused> = HashMap::new();
    for (voice_name, hits) in voices {
        if hits.is_empty() {
            continue;
        }
        let mut sorted = hits.clone();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (idx, hit) in sorted.iter().enumerate() {
            let rank = idx + 1;
            let contribution = rrf_contribution(rank);
            let entry = combined
                .entry(hit.memory_id.clone())
                .or_insert_with(|| Fused {
                    memory_id: hit.memory_id.clone(),
                    combined_score: 0.0,
                    voices: Vec::new(),
                });
            entry.combined_score += contribution;
            if !entry.voices.iter().any(|v| v == voice_name) {
                entry.voices.push((*voice_name).to_string());
            }
        }
    }
    let mut out: Vec<Fused> = combined.into_values().collect();
    out.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Diversity rerank (`polyphonic_recall.py` `_diversity_rerank` L737-L781): greedily keep the
/// highest-scoring candidates, dropping any whose voice-set Jaccard against an already-selected
/// candidate exceeds `0.8`.
pub fn diversity_rerank(fused: Vec<Fused>, top_k: usize) -> Vec<Fused> {
    let mut selected: Vec<Fused> = Vec::new();
    for cand in fused {
        if selected.len() >= top_k {
            break;
        }
        let diverse = selected
            .iter()
            .all(|sel| voice_jaccard(&cand.voices, &sel.voices) <= 0.8);
        if diverse {
            selected.push(cand);
        }
    }
    selected
}

/// Jaccard similarity of two voice-name sets (`_estimate_similarity` L769-L781).
fn voice_jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<&String> = a.iter().collect();
    let sb: std::collections::HashSet<&String> = b.iter().collect();
    sa.intersection(&sb).count() as f64 / sa.union(&sb).count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f64) -> VoiceHit {
        VoiceHit {
            memory_id: id.to_string(),
            score,
        }
    }

    #[test]
    fn rrf_rewards_cross_voice_agreement() {
        // "a" appears in two voices (rank 1 each); "b" only in one. "a" must win.
        let voices = vec![
            ("vector", vec![hit("a", 0.9), hit("b", 0.8)]),
            ("fact", vec![hit("a", 0.5)]),
        ];
        let fused = fuse(&voices);
        assert_eq!(fused[0].memory_id, "a");
        assert!(fused[0].combined_score > fused[1].combined_score);
        assert_eq!(fused[0].voices.len(), 2);
    }

    #[test]
    fn diversity_drops_same_voice_set() {
        let fused = vec![
            Fused {
                memory_id: "a".into(),
                combined_score: 0.5,
                voices: vec!["vector".into()],
            },
            Fused {
                memory_id: "b".into(),
                combined_score: 0.4,
                voices: vec!["vector".into()],
            },
        ];
        // Identical voice sets -> Jaccard 1.0 > 0.8 -> second dropped.
        let out = diversity_rerank(fused, 10);
        assert_eq!(out.len(), 1);
    }
}
