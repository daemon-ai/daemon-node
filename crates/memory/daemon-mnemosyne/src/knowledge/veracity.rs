// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Veracity consolidation — port of `veracity_consolidation.py`.
//!
//! `compute_fact_id` (length-prefixed NFC SHA-256, L111-L115) and the incremental confidence math
//! (L441, L536) are pure. The `consolidated_facts` upsert, `(S,P)` contradiction detection into
//! `conflicts`, and the higher-confidence-wins consolidation pass (`veracity_consolidation.py`
//! L460-L568 / `run_consolidation_pass`) are wired here over the SQLite tables.

use crate::error::Result;
use crate::util;
use rusqlite::{params, Connection, OptionalExtension};
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

/// The canonical veracity labels (`veracity_consolidation.py` `VERACITY_ALLOWED`).
const VERACITY_ALLOWED: &[&str] = &["stated", "inferred", "tool", "imported", "unknown"];

/// Cap on the raw value included in the clamp warning (`veracity_consolidation.py`
/// `_VERACITY_WARN_VALUE_CAP` L145): bounds log volume and content leakage from bad labels.
const VERACITY_WARN_VALUE_CAP: usize = 80;

/// Normalize and clamp a veracity label to the canonical allowlist (`veracity_consolidation.py`
/// `clamp_veracity` L148-L180): empty/whitespace clamps silently, non-canonical labels clamp to
/// `unknown` with a warning naming the calling `context`.
pub fn clamp_veracity(raw: &str, context: &str) -> String {
    let norm = raw.trim().to_lowercase();
    if norm.is_empty() {
        return "unknown".to_string();
    }
    if VERACITY_ALLOWED.contains(&norm.as_str()) {
        return norm;
    }
    let truncated: String = if raw.len() > VERACITY_WARN_VALUE_CAP {
        let cut: String = raw.chars().take(VERACITY_WARN_VALUE_CAP).collect();
        format!("{cut}...[truncated]")
    } else {
        raw.to_string()
    };
    tracing::warn!(context, raw = %truncated, "unknown veracity label; clamping to 'unknown'");
    "unknown".to_string()
}

/// Aggregate per-source veracity labels into one summary label (`veracity_consolidation.py`
/// `aggregate_veracity` L183-L244): drop non-canonical labels; treat `unknown` as low-priority (only
/// counted when no canonical non-`unknown` label is present); then take the mode, breaking multi-way
/// ties toward the lowest-weight (most conservative) label.
pub fn aggregate_veracity(source_veracities: &[String]) -> String {
    let valid: Vec<&str> = source_veracities
        .iter()
        .map(|s| s.as_str())
        .filter(|v| VERACITY_ALLOWED.contains(v))
        .collect();
    if valid.is_empty() {
        return "unknown".to_string();
    }
    let non_unknown: Vec<&str> = valid.iter().copied().filter(|v| *v != "unknown").collect();
    let candidates = if non_unknown.is_empty() {
        &valid
    } else {
        &non_unknown
    };

    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for v in candidates {
        *counts.entry(*v).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    let mut most_common: Vec<&str> = counts
        .iter()
        .filter(|(_, c)| **c == max_count)
        .map(|(v, _)| *v)
        .collect();
    if most_common.len() == 1 {
        return most_common[0].to_string();
    }
    // Tie: most conservative (lowest weight), deterministic by name on weight ties.
    most_common.sort_by(|a, b| {
        veracity_weight(a)
            .partial_cmp(&veracity_weight(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    most_common[0].to_string()
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

/// A consolidated fact and the effect of [`consolidate_fact`] on it.
#[derive(Clone, Debug)]
pub struct ConsolidatedFact {
    /// Deterministic id ([`compute_fact_id`]).
    pub id: String,
    /// Subject / predicate / object.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object.
    pub object: String,
    /// Confidence after the update.
    pub confidence: f64,
    /// Number of times this exact SPO has been seen.
    pub mention_count: i64,
}

/// Upsert a fact into `consolidated_facts` (`veracity_consolidation.py` `consolidate_fact`
/// L460-L568). An existing SPO has its confidence Bayesian-updated and mention count bumped; a new
/// SPO is inserted at the initial confidence and any same-`(subject, predicate)` rows with a
/// different object are recorded as `contradiction` rows in `conflicts`.
pub fn consolidate_fact(
    conn: &Connection,
    subject: &str,
    predicate: &str,
    object: &str,
    veracity: &str,
    source: &str,
) -> Result<ConsolidatedFact> {
    let now = util::now_iso();
    let existing: Option<(String, f64, i64, Option<String>)> = conn
        .query_row(
            "SELECT id, confidence, mention_count, sources_json FROM consolidated_facts \
             WHERE subject = ?1 AND predicate = ?2 AND object = ?3",
            params![subject, predicate, object],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, f64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()?;

    if let Some((id, confidence, mention_count, sources_json)) = existing {
        let new_confidence = bayesian_update(confidence, veracity);
        let new_count = mention_count + 1;
        let mut sources: Vec<String> =
            serde_json::from_str(sources_json.as_deref().unwrap_or("[]")).unwrap_or_default();
        if !source.is_empty() && !sources.iter().any(|s| s == source) {
            sources.push(source.to_string());
        }
        conn.execute(
            "UPDATE consolidated_facts \
             SET confidence = ?1, mention_count = ?2, last_seen = ?3, sources_json = ?4, \
                 veracity = ?5, updated_at = ?3 WHERE id = ?6",
            params![
                new_confidence,
                new_count,
                now,
                serde_json::to_string(&sources)?,
                veracity,
                id,
            ],
        )?;
        return Ok(ConsolidatedFact {
            id,
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence: new_confidence,
            mention_count: new_count,
        });
    }

    // New SPO: detect same-(subject, predicate) contradictions before inserting.
    let conflicts: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT id FROM consolidated_facts \
             WHERE subject = ?1 AND predicate = ?2 AND object != ?3",
        )?;
        let rows = stmt
            .query_map(params![subject, predicate, object], |r| {
                r.get::<_, String>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows
    };

    let fact_id = compute_fact_id(subject, predicate, object);
    let base_confidence = initial_confidence(veracity);
    let sources: Vec<String> = if source.is_empty() {
        Vec::new()
    } else {
        vec![source.to_string()]
    };
    conn.execute(
        "INSERT INTO consolidated_facts \
         (id, subject, predicate, object, confidence, mention_count, first_seen, last_seen, \
          sources_json, veracity) \
         VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6, ?6, ?7, ?8)",
        params![
            fact_id,
            subject,
            predicate,
            object,
            base_confidence,
            now,
            serde_json::to_string(&sources)?,
            veracity,
        ],
    )?;
    for conflict_id in conflicts {
        record_conflict(conn, &fact_id, &conflict_id, "contradiction")?;
    }
    Ok(ConsolidatedFact {
        id: fact_id,
        subject: subject.to_string(),
        predicate: predicate.to_string(),
        object: object.to_string(),
        confidence: base_confidence,
        mention_count: 1,
    })
}

/// Record a conflict between two facts (`veracity_consolidation.py` `_record_conflict`).
pub fn record_conflict(
    conn: &Connection,
    fact_a_id: &str,
    fact_b_id: &str,
    conflict_type: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?1, ?2, ?3)",
        params![fact_a_id, fact_b_id, conflict_type],
    )?;
    Ok(())
}

/// The number of unresolved conflict rows (test/diagnostic helper).
pub fn conflict_count(conn: &Connection) -> Result<usize> {
    let n: i64 = conn.query_row("SELECT COUNT(*) FROM conflicts", [], |r| r.get(0))?;
    Ok(n as usize)
}

/// Background consolidation pass (`veracity_consolidation.py` `run_consolidation_pass` L520+): for
/// each well-attested fact (`mention_count > 2`), any conflicting `(subject, predicate)` row with a
/// strictly lower confidence is auto-resolved by marking it `superseded_by` the winner. Returns the
/// number of facts superseded.
pub fn run_consolidation_pass(conn: &Connection) -> Result<usize> {
    let primary: Vec<(String, String, String, String, f64)> = {
        let mut stmt = conn.prepare(
            "SELECT id, subject, predicate, object, confidence FROM consolidated_facts \
             WHERE mention_count > 2 AND superseded_by IS NULL ORDER BY mention_count DESC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, f64>(4)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows
    };

    let now = util::now_iso();
    let mut resolved = 0usize;
    for (id, subject, predicate, object, confidence) in primary {
        let losers: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT id FROM consolidated_facts \
                 WHERE subject = ?1 AND predicate = ?2 AND object != ?3 AND superseded_by IS NULL \
                   AND confidence < ?4",
            )?;
            let rows = stmt
                .query_map(params![subject, predicate, object, confidence], |r| {
                    r.get::<_, String>(0)
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };
        for loser in losers {
            conn.execute(
                "UPDATE consolidated_facts SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
                params![id, now, loser],
            )?;
            resolved += 1;
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

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

    #[test]
    fn repeated_mention_bumps_confidence_and_count() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        let f1 = consolidate_fact(&c, "Maya", "works_at", "Acme", "stated", "m1").unwrap();
        assert_eq!(f1.mention_count, 1);
        let f2 = consolidate_fact(&c, "Maya", "works_at", "Acme", "stated", "m2").unwrap();
        assert_eq!(f2.mention_count, 2);
        assert!(f2.confidence > f1.confidence);
    }

    #[test]
    fn contradiction_records_conflict() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        consolidate_fact(&c, "Maya", "works_at", "Acme", "stated", "m1").unwrap();
        consolidate_fact(&c, "Maya", "works_at", "Globex", "stated", "m2").unwrap();
        assert_eq!(conflict_count(&c).unwrap(), 1);
    }

    #[test]
    fn consolidation_pass_supersedes_lower_confidence() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        // Acme seen 3x (mention_count > 2, higher confidence), Globex once.
        for src in ["m1", "m2", "m3"] {
            consolidate_fact(&c, "Maya", "works_at", "Acme", "stated", src).unwrap();
        }
        consolidate_fact(&c, "Maya", "works_at", "Globex", "inferred", "m4").unwrap();
        let resolved = run_consolidation_pass(&c).unwrap();
        assert_eq!(resolved, 1);
        let superseded: Option<String> = c
            .query_row(
                "SELECT superseded_by FROM consolidated_facts WHERE object = 'Globex'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(superseded.is_some());
    }
}
