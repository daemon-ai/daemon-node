//! The shared recall scoring math — port of the constants/formulas in `beam.py`.
//!
//! These pure functions are the heart of the port and are unit-tested for parity. See
//! `mnemosyne-rust-port-spec.md` §7 for the line references.

/// Default hybrid weights `(vec, fts, importance)` before normalization (`beam.py` L1157).
pub const DEFAULT_WEIGHTS: (f64, f64, f64) = (0.5, 0.3, 0.2);

/// Recency half-life in hours (`RECENCY_HALFLIFE_HOURS`, `beam.py` L1202).
pub const RECENCY_HALFLIFE_HOURS: f64 = 168.0;

/// Episodic tier weights `T1/T2/T3` (`beam.py` L5931).
pub const TIER_WEIGHTS: [f64; 3] = [1.0, 0.5, 0.25];

/// Normalize `(vec, fts, importance)` to sum 1.0, clamping negatives (`beam.py` L1157-L1183).
pub fn normalize_weights(vw: f64, fw: f64, iw: f64) -> (f64, f64, f64) {
    let vw = vw.max(0.0);
    let fw = fw.max(0.0);
    let iw = iw.max(0.0);
    let total = vw + fw + iw;
    if total <= 0.0 {
        return DEFAULT_WEIGHTS;
    }
    (vw / total, fw / total, iw / total)
}

/// Exponential recency decay `exp(-age_hours / halflife)`; unknown age -> 0.5 (`beam.py` L1202-L1214).
pub fn recency_decay(age_hours: Option<f64>) -> f64 {
    match age_hours {
        Some(age) => (-age / RECENCY_HALFLIFE_HOURS).exp(),
        None => 0.5,
    }
}

/// Lexical relevance floor by query token count (`beam.py` L1517-L1527).
pub fn lexical_floor(query_tokens: usize) -> f64 {
    match query_tokens {
        n if n >= 4 => 0.3,
        3 => 0.5,
        _ => 0.15,
    }
}

/// Veracity multiplier (`veracity_consolidation.py` `VERACITY_WEIGHTS`, applied at `beam.py` L5931).
pub fn veracity_multiplier(veracity: &str) -> f64 {
    crate::knowledge::veracity::veracity_weight(veracity)
}

/// The episodic hybrid score (`beam.py` L5720-L5793), excluding the candidate-drop gate (the caller
/// applies `lexical < floor && sim < 0.65 -> drop`).
///
/// `score = max(sim*vw + fts*fw + imp*iw, lexical*0.8) * (0.7 + 0.3*decay)`
/// then `+= graph_bonus + fact_bonus + binary_bonus`.
#[allow(clippy::too_many_arguments)]
pub fn episodic_score(
    sim: f64,
    fts: f64,
    importance: f64,
    lexical: f64,
    decay: f64,
    weights: (f64, f64, f64),
    graph_bonus: f64,
    fact_bonus: f64,
    binary_bonus: f64,
) -> f64 {
    let (vw, fw, iw) = weights;
    let base = sim * vw + fts * fw + importance * iw;
    let mut score = base.max(lexical * 0.8) * (0.7 + 0.3 * decay);
    score += graph_bonus + fact_bonus + binary_bonus;
    score
}

/// The working-memory score (`beam.py` L5314-L5328).
///
/// `base = relevance*kw_share + imp*iw + relevance^2*0.08`; if `vec_sim>0`, blend
/// `base*0.8 + vec_sim*0.2`; then `* (rc_share + (1-rc_share)*decay)`.
pub fn working_memory_score(
    relevance: f64,
    importance: f64,
    iw: f64,
    vec_sim: f64,
    decay: f64,
) -> f64 {
    let kw_share = (1.0 - iw) * 0.6;
    let rc_share = (1.0 - iw) * 0.4;
    let mut base = relevance * kw_share + importance * iw + relevance.powi(2) * 0.08;
    if vec_sim > 0.0 {
        base = base * 0.80 + vec_sim * 0.20;
    }
    base * (rc_share + (1.0 - rc_share) * decay)
}

/// Graph bonus `min(edge_count * 0.02, 0.08)` (`beam.py` L5779).
pub fn graph_bonus(edge_count: usize) -> f64 {
    (edge_count as f64 * 0.02).min(0.08)
}

/// Fact bonus `min(match_count * 0.04, 0.10)` (`beam.py` L5781).
pub fn fact_bonus(match_count: usize) -> f64 {
    (match_count as f64 * 0.04).min(0.10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_normalize_to_one() {
        let (v, f, i) = normalize_weights(0.5, 0.3, 0.2);
        assert!((v + f + i - 1.0).abs() < 1e-9);
        assert!((v - 0.5).abs() < 1e-9);
    }

    #[test]
    fn decay_bounds() {
        assert!((recency_decay(Some(0.0)) - 1.0).abs() < 1e-9);
        assert_eq!(recency_decay(None), 0.5);
        assert!(recency_decay(Some(168.0)) < 0.4); // exp(-1)
    }

    #[test]
    fn floors_match_python() {
        assert_eq!(lexical_floor(5), 0.3);
        assert_eq!(lexical_floor(3), 0.5);
        assert_eq!(lexical_floor(1), 0.15);
    }

    #[test]
    fn episodic_score_uses_lexical_fallback() {
        // Strong lexical, weak hybrid -> max() picks the lexical*0.8 branch.
        let s = episodic_score(0.0, 0.0, 0.0, 1.0, 1.0, DEFAULT_WEIGHTS, 0.0, 0.0, 0.0);
        assert!((s - 0.8).abs() < 1e-9);
    }
}
