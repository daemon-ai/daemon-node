//! Veracity consolidation — port of `veracity_consolidation.py`.
//!
//! `compute_fact_id` (length-prefixed NFC SHA-256, L111-L115) and the incremental confidence math
//! (L441, L536) are pure and ported here with tests. The `consolidated_facts` / `conflicts` storage
//! and `(S,P)` contradiction detection are TODO (scaffold).

use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

/// Per-source veracity weights (`veracity_consolidation.py` `VERACITY_WEIGHTS` L122-L128).
pub fn veracity_weight(veracity: &str) -> f64 {
    match veracity {
        "stated" => 1.0,
        "inferred" => 0.7,
        "tool" => 0.5,
        "imported" => 0.6,
        _ => 0.8, // unknown / anything else
    }
}

/// Deterministic fact id `cf_<sha256(len-prefixed NFC SPO)[:24]>` (`veracity_consolidation.py`
/// L111-L115). Length-prefix framing prevents separator smuggling; NFC for Unicode stability.
pub fn compute_fact_id(subject: &str, predicate: &str, object: &str) -> String {
    let mut hasher = Sha256::new();
    for value in [subject, predicate, object] {
        let normalized: String = value.nfc().collect();
        let bytes = normalized.as_bytes();
        hasher.update(format!("{}:", bytes.len()).as_bytes());
        hasher.update(bytes);
    }
    let hex = format!("{:x}", hasher.finalize());
    format!("cf_{}", &hex[..24])
}

/// The base confidence for a brand-new fact: `weight * 0.5` (`veracity_consolidation.py` L536).
pub fn initial_confidence(veracity: &str) -> f64 {
    veracity_weight(veracity) * 0.5
}

/// Incremental Bayesian-ish update for a repeated mention:
/// `min(old + (1 - old) * weight * 0.3, 1.0)` (`veracity_consolidation.py` `bayesian_update` L441).
pub fn bayesian_update(current_confidence: f64, veracity: &str) -> f64 {
    let weight = veracity_weight(veracity);
    let increment = (1.0 - current_confidence) * weight * 0.3;
    (current_confidence + increment).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_id_is_stable_and_prefixed() {
        let a = compute_fact_id("Maya", "assigned_to", "auth");
        let b = compute_fact_id("Maya", "assigned_to", "auth");
        assert_eq!(a, b);
        assert!(a.starts_with("cf_"));
        assert_eq!(a.len(), 3 + 24);
    }

    #[test]
    fn confidence_sequence_matches_python() {
        // stated weight = 1.0: 0.5 -> 0.65 -> 0.755 -> 0.8285 ...
        let mut c = initial_confidence("stated");
        assert!((c - 0.5).abs() < 1e-9);
        c = bayesian_update(c, "stated");
        assert!((c - 0.65).abs() < 1e-9);
        c = bayesian_update(c, "stated");
        assert!((c - 0.755).abs() < 1e-9);
        c = bayesian_update(c, "stated");
        assert!((c - 0.8285).abs() < 1e-9);
    }

    #[test]
    fn confidence_caps_at_one() {
        let mut c = 0.99;
        for _ in 0..50 {
            c = bayesian_update(c, "stated");
        }
        assert!(c <= 1.0);
    }
}
