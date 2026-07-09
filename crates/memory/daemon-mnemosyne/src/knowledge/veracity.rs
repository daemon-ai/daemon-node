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

/// Run `f` inside a `BEGIN IMMEDIATE` transaction, or inside the caller's transaction when one
/// is already open (`veracity_consolidation.py` `_serialized_write`, E2.a.5/E2.a.6). SELECT-then-
/// write sequences on `consolidated_facts` are path-dependent; without writer serialization two
/// connections both pass the no-match SELECT and race the INSERT (silent data loss via the
/// deterministic PRIMARY KEY), and concurrent Bayesian updates overwrite instead of compounding.
/// On error the owned transaction rolls back (drop); a caller-owned transaction is left untouched.
pub fn serialized_write<T>(
    conn: &Connection,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    if !conn.is_autocommit() {
        // Nested call: the caller owns the transaction lifecycle (`conn.in_transaction` check).
        return f(conn);
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    let out = f(&tx)?;
    tx.commit()?;
    Ok(out)
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
/// different object are recorded as `contradiction` rows in `conflicts`. The whole SELECT-then-
/// write sequence runs under [`serialized_write`] so concurrent same-SPO callers serialize
/// instead of racing (E2.a.5); conflict rows commit atomically with the fact insert.
pub fn consolidate_fact(
    conn: &Connection,
    subject: &str,
    predicate: &str,
    object: &str,
    veracity: &str,
    source: &str,
) -> Result<ConsolidatedFact> {
    serialized_write(conn, |conn| {
        consolidate_fact_locked(conn, subject, predicate, object, veracity, source)
    })
}

/// The [`consolidate_fact`] body, running inside the serialized-write scope.
fn consolidate_fact_locked(
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

/// Read consolidated facts, optionally filtered by exact subject, above a confidence floor,
/// excluding superseded rows (`veracity_consolidation.py` `get_consolidated_facts` L706-L752).
/// Ordered by confidence then mention count, both descending.
pub fn get_consolidated_facts(
    conn: &Connection,
    subject: Option<&str>,
    min_confidence: f64,
) -> Result<Vec<ConsolidatedFact>> {
    let base = "SELECT id, subject, predicate, object, confidence, mention_count \
                FROM consolidated_facts";
    let order = "ORDER BY confidence DESC, mention_count DESC";
    let map = |r: &rusqlite::Row<'_>| -> rusqlite::Result<ConsolidatedFact> {
        Ok(ConsolidatedFact {
            id: r.get(0)?,
            subject: r.get(1)?,
            predicate: r.get(2)?,
            object: r.get(3)?,
            confidence: r.get(4)?,
            mention_count: r.get(5)?,
        })
    };
    let rows = if let Some(subject) = subject {
        let mut stmt = conn.prepare(&format!(
            "{base} WHERE subject = ?1 AND confidence >= ?2 AND superseded_by IS NULL {order}"
        ))?;
        let rows = stmt
            .query_map(params![subject, min_confidence], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows
    } else {
        let mut stmt = conn.prepare(&format!(
            "{base} WHERE confidence >= ?1 AND superseded_by IS NULL {order}"
        ))?;
        let rows = stmt
            .query_map(params![min_confidence], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows
    };
    Ok(rows)
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
/// strictly lower confidence is auto-resolved by marking it `superseded_by` the winner. Runs under
/// [`serialized_write`] (E2.a.6) so interleaved writers can't mutate the pass's read-decide-resolve
/// loop mid-flight. Returns the number of facts superseded.
pub fn run_consolidation_pass(conn: &Connection) -> Result<usize> {
    serialized_write(conn, run_consolidation_pass_locked)
}

/// The [`run_consolidation_pass`] body, running inside the serialized-write scope.
fn run_consolidation_pass_locked(conn: &Connection) -> Result<usize> {
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

    // parity: test_consolidate_fact_id_collision.py::test_compute_fact_id_distinguishes_distinct_spos (tests/test_consolidate_fact_id_collision.py:61)
    #[test]
    fn fact_id_distinguishes_distinct_spos() {
        let ids = [
            compute_fact_id("Alice", "is", "developer"),
            compute_fact_id("Bob", "is", "developer"),
            compute_fact_id("Alice", "owns", "developer"),
            compute_fact_id("Alice", "is", "manager"),
        ];
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), 4, "distinct SPOs must produce distinct ids");
    }

    // parity: test_consolidate_fact_id_collision.py::test_compute_fact_id_long_content_does_not_collide (tests/test_consolidate_fact_id_collision.py:82)
    #[test]
    fn fact_id_long_content_does_not_collide() {
        let subject_a = "Alice Anderson the Senior Staff Engineer responsible for the \
                         authentication subsystem";
        let subject_b = "Alice Anderson the Senior Staff Engineer responsible for the \
                         authorization subsystem";
        let predicate = "is_described_in_the_internal_documentation_at_section_4_paragraph_3_as";
        let object = "a competent and reliable engineer";
        assert_ne!(
            compute_fact_id(subject_a, predicate, object),
            compute_fact_id(subject_b, predicate, object),
            "long SPOs must hash the full input (the pre-fix truncated f-string collided)"
        );
    }

    // parity: test_consolidate_fact_id_collision.py::test_compute_fact_id_separator_prevents_smuggling (tests/test_consolidate_fact_id_collision.py:104)
    // parity: test_consolidate_fact_id_collision.py::TestReviewHardening::test_separator_smuggling_does_not_collide (tests/test_consolidate_fact_id_collision.py:292)
    #[test]
    fn fact_id_length_prefix_prevents_separator_smuggling() {
        assert_ne!(
            compute_fact_id("a_b", "c", "d"),
            compute_fact_id("a", "b_c", "d"),
            "underscore-joined forms must not collide"
        );
        assert_ne!(
            compute_fact_id("a\u{1f}", "b", "c"),
            compute_fact_id("a", "\u{1f}b", "c"),
            "embedded separators must not collide under length-prefix framing"
        );
    }

    // parity: test_consolidate_fact_id_collision.py::TestReviewHardening::test_unicode_nfc_and_nfd_hash_identically (tests/test_consolidate_fact_id_collision.py:304)
    #[test]
    fn fact_id_nfc_and_nfd_hash_identically() {
        let nfc = "prot\u{e9}g\u{e9}"; // precomposed é
        let nfd = "prote\u{301}ge\u{301}"; // e + combining acute
        assert_ne!(nfc, nfd, "test setup: NFC/NFD strings must differ bytewise");
        assert_eq!(
            compute_fact_id(nfc, "is", "mentored"),
            compute_fact_id(nfd, "is", "mentored"),
            "NFC normalization must be applied before hashing"
        );
    }

    // parity: test_consolidate_fact_id_collision.py::TestReviewHardening::test_hash_uses_sha256_codebase_consistency (tests/test_consolidate_fact_id_collision.py:337)
    #[test]
    fn fact_id_pins_sha256_of_length_prefixed_nfc() {
        // SHA-256 of "5:Alice2:is9:developer" (byte-length-prefixed NFC components).
        let mut hasher = Sha256::new();
        hasher.update(b"5:Alice2:is9:developer");
        let expected = format!("cf_{}", &format!("{:x}", hasher.finalize())[..24]);
        assert_eq!(compute_fact_id("Alice", "is", "developer"), expected);
    }

    // parity: test_consolidate_fact_id_collision.py::test_consolidate_fact_stores_hash_based_id (tests/test_consolidate_fact_id_collision.py:123)
    #[test]
    fn consolidate_stores_the_computed_hash_id() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        consolidate_fact(&c, "Carol", "leads", "the platform team", "stated", "m1").unwrap();
        let stored: String = c
            .query_row(
                "SELECT id FROM consolidated_facts WHERE subject = 'Carol'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            stored,
            compute_fact_id("Carol", "leads", "the platform team")
        );
        assert!(stored.starts_with("cf_") && stored.len() == 27);
    }

    // parity: test_consolidate_fact_id_collision.py::test_consolidate_fact_distinct_long_content_both_stored (tests/test_consolidate_fact_id_collision.py:154)
    #[test]
    fn distinct_long_content_facts_both_stored() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        let pred = "is_described_in_the_internal_documentation_at_section_4_paragraph_3_as";
        consolidate_fact(
            &c,
            "EngineerLeadAlice",
            pred,
            "a competent and reliable engineer who delivers on time",
            "stated",
            "mem_x",
        )
        .unwrap();
        consolidate_fact(
            &c,
            "EngineerLeadAlice",
            pred,
            "a competent and reliable engineer who escalates blockers",
            "stated",
            "mem_y",
        )
        .unwrap();
        let ids: Vec<String> = {
            let mut stmt = c
                .prepare("SELECT id FROM consolidated_facts WHERE subject = 'EngineerLeadAlice'")
                .unwrap();
            let rows = stmt.query_map([], |r| r.get(0)).unwrap();
            rows.collect::<std::result::Result<Vec<_>, _>>().unwrap()
        };
        assert_eq!(ids.len(), 2, "both long facts stored");
        assert_ne!(ids[0], ids[1], "distinct facts share an id");
    }

    // parity: test_consolidate_fact_id_collision.py::test_mixed_format_db_dedup_still_finds_old_rows (tests/test_consolidate_fact_id_collision.py:185)
    #[test]
    fn legacy_format_rows_dedup_by_spo_and_keep_their_id() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        let legacy_id = "cf_Eve_is_a_lawyer";
        c.execute(
            "INSERT INTO consolidated_facts \
             (id, subject, predicate, object, confidence, mention_count, first_seen, last_seen, \
              sources_json, veracity) \
             VALUES (?1, 'Eve', 'is', 'a lawyer', 0.5, 1, '2026-01-01T00:00:00', \
                     '2026-01-01T00:00:00', '[]', 'stated')",
            params![legacy_id],
        )
        .unwrap();

        // Dedup matches on SPO, not id: the legacy row is UPDATED, its id preserved.
        let fact = consolidate_fact(&c, "Eve", "is", "a lawyer", "stated", "mem_new").unwrap();
        assert_eq!(fact.id, legacy_id, "legacy id preserved on the update path");
        let (rows, count): (i64, i64) = c
            .query_row(
                "SELECT COUNT(*), MAX(mention_count) FROM consolidated_facts \
                 WHERE subject = 'Eve'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(rows, 1, "legacy row not matched — duplicate created");
        assert_eq!(count, 2);
    }

    /// Open one raw connection to a bank file, configured like a Python thread worker
    /// (own connection + busy_timeout, `test_consolidate_fact_concurrency.py:69`).
    fn worker_conn(path: &std::path::Path) -> Connection {
        let conn = Connection::open(path).expect("open worker connection");
        conn.busy_timeout(std::time::Duration::from_millis(5000))
            .expect("busy_timeout");
        conn
    }

    // parity: test_consolidate_fact_concurrency.py::test_eight_threads_same_spo_produce_one_row_count_8 (tests/test_consolidate_fact_concurrency.py:123)
    // parity: test_consolidate_fact_concurrency.py::test_two_threads_same_spo_produce_one_row_count_2 (tests/test_consolidate_fact_concurrency.py:85)
    #[test]
    fn concurrent_same_spo_yields_one_row_with_all_mentions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrency.db");
        drop(Store::open(&path).unwrap()); // initialize the schema once (WAL persists)

        let barrier = std::sync::Barrier::new(8);
        let errors: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let barrier = &barrier;
                    let path = &path;
                    s.spawn(move || -> std::result::Result<(), String> {
                        let conn = worker_conn(path);
                        barrier.wait(); // maximize contention
                        consolidate_fact(
                            &conn,
                            "Carol",
                            "leads",
                            "team",
                            "stated",
                            &format!("src_{i}"),
                        )
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("thread panicked").err())
                .collect()
        });
        assert!(
            errors.is_empty(),
            "threads raised under contention (observations silently lost): {errors:?}"
        );

        let conn = worker_conn(&path);
        let rows: Vec<i64> = {
            let mut stmt = conn
                .prepare("SELECT mention_count FROM consolidated_facts WHERE subject = 'Carol'")
                .unwrap();
            let rows = stmt.query_map([], |r| r.get(0)).unwrap();
            rows.collect::<std::result::Result<Vec<_>, _>>().unwrap()
        };
        assert_eq!(rows.len(), 1, "race produced duplicate rows: {rows:?}");
        assert_eq!(
            rows[0], 8,
            "expected mention_count 8 — race lost observations"
        );
    }

    // parity: test_consolidate_fact_concurrency.py::test_concurrent_updates_compound_confidence_correctly (tests/test_consolidate_fact_concurrency.py:261)
    #[test]
    fn concurrent_updates_compound_confidence_and_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compound.db");
        drop(Store::open(&path).unwrap());
        let seed_confidence = {
            let conn = worker_conn(&path);
            consolidate_fact(&conn, "Frank", "is", "DBA", "stated", "src_seed")
                .unwrap()
                .confidence
        };

        let barrier = std::sync::Barrier::new(4);
        let errors: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let barrier = &barrier;
                    let path = &path;
                    s.spawn(move || -> std::result::Result<(), String> {
                        let conn = worker_conn(path);
                        barrier.wait();
                        consolidate_fact(&conn, "Frank", "is", "DBA", "stated", &format!("src_{i}"))
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("thread panicked").err())
                .collect()
        });
        assert!(errors.is_empty(), "threads raised: {errors:?}");

        let conn = worker_conn(&path);
        let (count, confidence): (i64, f64) = conn
            .query_row(
                "SELECT mention_count, confidence FROM consolidated_facts \
                 WHERE subject = 'Frank'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        // Each update is path-dependent (`new = old + (1-old)*w*0.3`): unserialized concurrent
        // read-compute-write cycles overwrite each other instead of compounding.
        assert_eq!(
            count, 5,
            "expected mention_count 5 (seed + 4 concurrent updates) — race lost update(s)"
        );
        assert!(
            confidence > seed_confidence,
            "confidence {confidence} did not compound above the seed {seed_confidence}"
        );
    }

    // parity: test_consolidate_fact_concurrency.py::test_consolidate_fact_nested_in_outer_transaction (tests/test_consolidate_fact_concurrency.py:192)
    // parity: test_consolidate_fact_sibling_races.py::test_serialized_write_participates_in_outer_transaction (tests/test_consolidate_fact_sibling_races.py:299)
    #[test]
    fn consolidate_fact_participates_in_outer_transaction() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        c.execute_batch("BEGIN IMMEDIATE;").unwrap();
        assert!(!c.is_autocommit(), "test setup: outer transaction open");

        // Must NOT raise "cannot start a transaction within a transaction" and must NOT commit
        // the caller's transaction.
        let fact = consolidate_fact(&c, "Dan", "is", "designer", "stated", "src_x").unwrap();
        assert_eq!(fact.subject, "Dan");
        assert!(
            !c.is_autocommit(),
            "the nested call must leave the outer transaction open"
        );

        c.execute_batch("COMMIT;").unwrap();
        let count: i64 = c
            .query_row(
                "SELECT mention_count FROM consolidated_facts WHERE subject = 'Dan'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "the row persists once the caller commits");
    }

    // parity: test_consolidate_fact_sibling_races.py::test_serialized_write_begins_immediate_when_not_in_tx (tests/test_consolidate_fact_sibling_races.py:246)
    // parity: test_consolidate_fact_sibling_races.py::test_serialized_write_rolls_back_on_exception (tests/test_consolidate_fact_sibling_races.py:272)
    #[test]
    fn serialized_write_owns_commit_and_rolls_back_on_error() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();

        // Happy path: the helper opens its own transaction and commits it.
        serialized_write(&c, |conn| {
            assert!(!conn.is_autocommit(), "helper must open a transaction");
            conn.execute(
                "INSERT INTO consolidated_facts \
                 (id, subject, predicate, object, confidence, mention_count, first_seen, \
                  last_seen, sources_json, veracity) \
                 VALUES ('cf_test', 's', 'p', 'o', 0.5, 1, datetime('now'), datetime('now'), \
                         '[]', 'stated')",
                [],
            )?;
            Ok(())
        })
        .unwrap();
        assert!(c.is_autocommit(), "helper must close its transaction");
        let stored: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM consolidated_facts WHERE id = 'cf_test'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, 1);

        // Error path: the body's writes roll back.
        let res: Result<()> = serialized_write(&c, |conn| {
            conn.execute(
                "INSERT INTO consolidated_facts \
                 (id, subject, predicate, object, confidence, mention_count, first_seen, \
                  last_seen, sources_json, veracity) \
                 VALUES ('cf_doomed', 's', 'p', 'o', 0.5, 1, datetime('now'), datetime('now'), \
                         '[]', 'stated')",
                [],
            )?;
            Err(crate::error::Error::Invalid(
                "simulated mid-write failure".to_string(),
            ))
        });
        assert!(res.is_err());
        assert!(c.is_autocommit(), "transaction closed after rollback");
        let leaked: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM consolidated_facts WHERE id = 'cf_doomed'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leaked, 0, "rollback didn't undo the insert");
    }

    // parity: test_consolidate_fact_concurrency.py::test_eight_threads_distinct_spos_produce_eight_rows (tests/test_consolidate_fact_concurrency.py:157)
    #[test]
    fn concurrent_distinct_spos_all_stored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("distinct.db");
        drop(Store::open(&path).unwrap());

        let barrier = std::sync::Barrier::new(8);
        let errors: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let barrier = &barrier;
                    let path = &path;
                    s.spawn(move || -> std::result::Result<(), String> {
                        let conn = worker_conn(path);
                        barrier.wait();
                        consolidate_fact(
                            &conn,
                            &format!("Person{i}"),
                            "is",
                            "engineer",
                            "stated",
                            &format!("src_{i}"),
                        )
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("thread panicked").err())
                .collect()
        });
        assert!(errors.is_empty(), "threads raised: {errors:?}");

        let conn = worker_conn(&path);
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM consolidated_facts WHERE subject LIKE 'Person%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 8, "all 8 distinct SPOs must land");
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
