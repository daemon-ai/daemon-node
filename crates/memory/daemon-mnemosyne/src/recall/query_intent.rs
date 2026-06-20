//! Query-intent classification + weight bias — port of `query_intent.py`.
//!
//! Per-category weight biases (`INTENT_WEIGHTS` L86-L93) applied as `base*bias` then renormalized
//! (`adjust_weights` L137-L167). Scaffold: the regex `INTENT_PATTERNS` (L41-L82) are TODO; the bias
//! table + renormalization are ported.

/// A classified query intent category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// Time-oriented query.
    Temporal,
    /// Fact lookup.
    Factual,
    /// Entity-centric.
    Entity,
    /// Preference recall.
    Preference,
    /// How-to / procedural.
    Procedural,
    /// Default.
    General,
}

/// `(vec_bias, fts_bias, importance_bias)` per intent (`query_intent.py` `INTENT_WEIGHTS` L86-L93).
pub fn intent_bias(intent: Intent) -> (f64, f64, f64) {
    match intent {
        Intent::Temporal => (0.6, 1.5, 0.8),
        Intent::Factual => (1.0, 1.2, 0.9),
        Intent::Entity => (1.1, 1.0, 1.3),
        Intent::Preference => (0.9, 0.8, 1.5),
        Intent::Procedural => (1.3, 0.9, 0.7),
        Intent::General => (1.0, 1.0, 1.0),
    }
}

/// Apply the intent bias to base weights and renormalize (`query_intent.py` `adjust_weights`
/// L137-L167).
pub fn adjust_weights(base: (f64, f64, f64), intent: Intent) -> (f64, f64, f64) {
    let (bv, bf, bi) = intent_bias(intent);
    let vw = base.0 * bv;
    let fw = base.1 * bf;
    let iw = base.2 * bi;
    let total = vw + fw + iw;
    if total > 0.0 {
        (vw / total, fw / total, iw / total)
    } else {
        base
    }
}

/// Classify a query into an [`Intent`] (`query_intent.py` `classify_intent` L96-L134). Scaffold:
/// always `General` until the regex patterns are ported.
pub fn classify_intent(_query: &str) -> Intent {
    Intent::General
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_biases_toward_fts() {
        let (_v, f, _i) = adjust_weights((0.5, 0.3, 0.2), Intent::Temporal);
        let (_v2, f2, _i2) = adjust_weights((0.5, 0.3, 0.2), Intent::General);
        assert!(f > f2); // temporal pushes weight toward FTS
    }
}
