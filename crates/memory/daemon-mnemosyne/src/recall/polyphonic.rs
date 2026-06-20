//! Polyphonic recall — port of `polyphonic_recall.py` (P2, `MNEMOSYNE_POLYPHONIC_RECALL=1`).
//!
//! Four voices (vector/graph/fact/temporal) fused by Reciprocal Rank Fusion. Scaffold: the RRF
//! constant + fusion helper are ported; the voices are TODO.

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
