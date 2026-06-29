// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Query-intent classification + weight bias — port of `query_intent.py`.
//!
//! The regex `INTENT_PATTERNS` (L41-L82) classify a query, the per-category `INTENT_WEIGHTS`
//! (L86-L93) bias the base `(vec, fts, importance)` weights, and `adjust_weights` (L137-L167)
//! renormalizes them to sum 1.0.

use regex::Regex;
use std::sync::OnceLock;

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

/// The regex `INTENT_PATTERNS` table (`query_intent.py` L41-L82), compiled once per category. The
/// category order matches the Python list so ties resolve identically (first-seen wins).
fn intent_patterns() -> &'static [(Intent, Vec<Regex>)] {
    static PATTERNS: OnceLock<Vec<(Intent, Vec<Regex>)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let raw: &[(Intent, &[&str])] = &[
            (
                Intent::Temporal,
                &[
                    r"\b(when|last|yesterday|today|tomorrow|ago|before|after|since|until|during|recently|lately)\b",
                    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    r"\b\d{4}-\d{2}-\d{2}\b",
                    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                    r"\b(this|next|last)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                    r"\b\d+\s+(day|week|month|year|hour|minute)s?\s+(ago|from now|later|earlier)\b",
                ],
            ),
            (
                Intent::Factual,
                &[
                    r"\bwhat\s+is\b",
                    r"\bwho\s+is\b",
                    r"\bwhere\s+is\b",
                    r"\b(definition|define|explain|meaning)\b",
                    r"\bhow\s+(many|much|long|far)\b",
                ],
            ),
            (
                Intent::Entity,
                &[
                    r"\b(tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\b",
                    r"\b(who\s+is|what\s+does)\s+[a-z]+\b",
                    r"\b(about|regarding|concerning)\s+[a-z]+\b",
                ],
            ),
            (
                Intent::Preference,
                &[
                    r"\b(prefer|like|dislike|want|hate|love|enjoy|favorite|best|worst)\b",
                    r"\b(should\s+i|would\s+you|do\s+you\s+recommend)\b",
                    r"\b(choose|pick|select|option|choice|decide)\b",
                ],
            ),
            (
                Intent::Procedural,
                &[
                    r"\bhow\s+(to|do|can|should|would)\b",
                    r"\b(step|process|procedure|workflow|guide|tutorial)\b",
                    r"\b(setup|install|configure|build|deploy|run|execute|start|stop)\b",
                ],
            ),
        ];
        raw.iter()
            .map(|(intent, pats)| {
                let res = pats
                    .iter()
                    .map(|p| Regex::new(p).expect("INTENT_PATTERNS regex must compile"))
                    .collect();
                (*intent, res)
            })
            .collect()
    })
}

/// Classify a query into an [`Intent`] (`query_intent.py` `classify_intent` L96-L134): each category
/// scores `min(0.3 + matches*0.15, 1.0)`; the highest-scoring category wins, defaulting to
/// `General`.
pub fn classify_intent(query: &str) -> Intent {
    let lower = query.to_lowercase();
    let mut best = Intent::General;
    let mut best_score = 0.0_f64;
    for (intent, patterns) in intent_patterns() {
        let matches = patterns.iter().filter(|re| re.is_match(&lower)).count();
        if matches > 0 {
            let score = (0.3 + matches as f64 * 0.15).min(1.0);
            if score > best_score {
                best_score = score;
                best = *intent;
            }
        }
    }
    best
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

    #[test]
    fn classifies_representative_queries() {
        assert_eq!(
            classify_intent("what happened last Monday"),
            Intent::Temporal
        );
        assert_eq!(
            classify_intent("what is the database password"),
            Intent::Factual
        );
        assert_eq!(
            classify_intent("how do I deploy the service"),
            Intent::Procedural
        );
        assert_eq!(
            classify_intent("which option should I pick"),
            Intent::Preference
        );
        assert_eq!(classify_intent("banana"), Intent::General);
    }
}
