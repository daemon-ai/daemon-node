// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! AnnotationStore — port of `annotations.py` (the E6 triplestore split).
//!
//! Append-only, multi-valued per-memory tags with a `(memory_id, kind, value)` unique index and
//! `INSERT OR IGNORE` writes (`annotations.py` L128-L264). Reads can filter `mentions` noise.

use crate::error::Result;
use rusqlite::{params, Connection};

/// The annotation kinds (`annotations.py` `ANNOTATION_KINDS` L77-L82).
pub const ANNOTATION_KINDS: &[&str] = &["mentions", "fact", "occurred_on", "has_source"];

/// Minimum fact length kept by [`filter_facts`] (`annotations.py` `MIN_FACT_LENGTH` L89).
pub const MIN_FACT_LENGTH: usize = 10;

/// Drop empty / too-short candidate facts (`annotations.py` `filter_facts` L92-L97: keeps
/// `len(f) > MIN_FACT_LENGTH`, counted in characters like Python). Applied by extraction call
/// sites before writing `fact` annotations so the threshold lives in one place.
pub fn filter_facts(facts: &[String]) -> Vec<String> {
    facts
        .iter()
        .filter(|f| f.chars().count() > MIN_FACT_LENGTH)
        .cloned()
        .collect()
}

/// One annotation row.
#[derive(Clone, Debug)]
pub struct Annotation {
    /// The annotated memory id.
    pub memory_id: String,
    /// Kind (`mentions`/`fact`/`occurred_on`/`has_source`).
    pub kind: String,
    /// Value (entity name, fact text, ...).
    pub value: String,
    /// Source tag.
    pub source: String,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
}

/// True if a `mentions` value is meta/system noise that should not surface as an entity
/// (`annotations.py` mentions noise filter). Reuses the entity-extraction stopword set.
fn is_mentions_noise(value: &str) -> bool {
    super::entities::is_stop_word(&value.to_lowercase())
}

/// Append an annotation (`annotations.py` `add` L208-L230). `INSERT OR IGNORE` against the
/// `(memory_id, kind, value)` unique index dedups repeats. An empty `source` is stored as NULL
/// (Python's `source=None` default). Returns the new row id (0 if ignored).
pub fn add(
    conn: &Connection,
    memory_id: &str,
    kind: &str,
    value: &str,
    source: &str,
    confidence: f64,
) -> Result<i64> {
    let source = if source.is_empty() {
        None
    } else {
        Some(source)
    };
    conn.execute(
        "INSERT OR IGNORE INTO annotations (memory_id, kind, value, source, confidence) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![memory_id, kind, value, source, confidence],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Batch-insert multiple values under one `(memory_id, kind)` (`annotations.py` `add_many`
/// L232-L264). Blank values are skipped. Returns the count of (attempted) rows.
pub fn add_many(
    conn: &Connection,
    memory_id: &str,
    kind: &str,
    values: &[String],
    source: &str,
    confidence: f64,
) -> Result<usize> {
    let mut n = 0usize;
    for value in values {
        if value.trim().is_empty() {
            continue;
        }
        add(conn, memory_id, kind, value, source, confidence)?;
        n += 1;
    }
    Ok(n)
}

/// All annotations for a memory, optionally filtered by kind (`annotations.py` `query_by_memory`).
pub fn query_by_memory(
    conn: &Connection,
    memory_id: &str,
    kind: Option<&str>,
) -> Result<Vec<Annotation>> {
    let (sql, has_kind) = match kind {
        Some(_) => (
            "SELECT memory_id, kind, value, source, confidence FROM annotations \
             WHERE memory_id = ?1 AND kind = ?2 ORDER BY created_at ASC, id ASC",
            true,
        ),
        None => (
            "SELECT memory_id, kind, value, source, confidence FROM annotations \
             WHERE memory_id = ?1 ORDER BY created_at ASC, id ASC",
            false,
        ),
    };
    let mut stmt = conn.prepare(sql)?;
    let map = |r: &rusqlite::Row<'_>| {
        Ok(Annotation {
            memory_id: r.get(0)?,
            kind: r.get(1)?,
            value: r.get(2)?,
            source: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            confidence: r.get::<_, Option<f64>>(4)?.unwrap_or(1.0),
        })
    };
    let rows = if has_kind {
        stmt.query_map(params![memory_id, kind.unwrap()], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        stmt.query_map(params![memory_id], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    Ok(rows)
}

/// A full annotation row for cross-store transfer (`annotations.py` `export_all` L266+): carries the
/// primary-key `id` so [`import_all`] can dedup by identity across databases.
#[derive(Clone, Debug)]
pub struct AnnotationExport {
    /// The primary-key row id.
    pub id: i64,
    /// The annotated memory id.
    pub memory_id: String,
    /// Kind.
    pub kind: String,
    /// Value.
    pub value: String,
    /// Source tag (`None` when unset).
    pub source: Option<String>,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
}

/// Export every annotation row, id-carrying, insertion-ordered (`annotations.py` `export_all`).
pub fn export_all(_conn: &Connection) -> Result<Vec<AnnotationExport>> {
    // Stub: real implementation lands in the green commit.
    Ok(Vec::new())
}

/// Import annotation rows preserving their ids, deduping by id (`annotations.py` `import_all`).
/// Returns `(inserted, skipped)`.
pub fn import_all(_conn: &Connection, _rows: &[AnnotationExport]) -> Result<(usize, usize)> {
    // Stub: real implementation lands in the green commit.
    Ok((0, 0))
}

/// All annotations of a kind, optionally filtered by value (`annotations.py` `query_by_kind`). When
/// `filter_noise` and `kind == "mentions"`, meta/system noise values are excluded.
pub fn query_by_kind(
    conn: &Connection,
    kind: &str,
    value: Option<&str>,
    filter_noise: bool,
) -> Result<Vec<Annotation>> {
    let (sql, has_value) = match value {
        Some(_) => (
            "SELECT memory_id, kind, value, source, confidence FROM annotations \
             WHERE kind = ?1 AND value = ?2 ORDER BY created_at ASC, id ASC",
            true,
        ),
        None => (
            "SELECT memory_id, kind, value, source, confidence FROM annotations \
             WHERE kind = ?1 ORDER BY created_at ASC, id ASC",
            false,
        ),
    };
    let mut stmt = conn.prepare(sql)?;
    let map = |r: &rusqlite::Row<'_>| {
        Ok(Annotation {
            memory_id: r.get(0)?,
            kind: r.get(1)?,
            value: r.get(2)?,
            source: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            confidence: r.get::<_, Option<f64>>(4)?.unwrap_or(1.0),
        })
    };
    let mut rows = if has_value {
        stmt.query_map(params![kind, value.unwrap()], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        stmt.query_map(params![kind], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    if filter_noise && kind == "mentions" {
        rows.retain(|a| !is_mentions_noise(&a.value));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn insert_or_ignore_dedups_repeats() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        add(&c, "m1", "mentions", "Acme", "regex", 1.0).unwrap();
        add(&c, "m1", "mentions", "Acme", "regex", 1.0).unwrap();
        let rows = query_by_memory(&c, "m1", Some("mentions")).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn filter_facts_drops_short_candidates() {
        let facts = vec![
            "short".to_string(),
            "exactly ten".to_string(), // 11 chars — kept
            "0123456789".to_string(),  // exactly 10 — dropped (strict >)
            String::new(),             // empty — dropped
            "Maya works at Acme Corp".to_string(),
        ];
        let kept = filter_facts(&facts);
        assert_eq!(kept, vec!["exactly ten", "Maya works at Acme Corp"]);
    }

    #[test]
    fn mentions_noise_filtered_on_read() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        add(&c, "m1", "mentions", "Acme", "regex", 1.0).unwrap();
        add(&c, "m1", "mentions", "System", "regex", 1.0).unwrap();
        let kept = query_by_kind(&c, "mentions", None, true).unwrap();
        assert!(kept.iter().any(|a| a.value == "Acme"));
        assert!(!kept.iter().any(|a| a.value == "System"));
    }

    // PARITY: Mnemosyne tests/test_annotations.py::TestAnnotationStoreMultiValuePreservation::test_multiple_mentions_for_one_memory_preserved
    // The E6 contract: multiple values under one (memory_id, kind) are append-only — no sibling
    // auto-invalidation (the TripleStore bug this store fixes).
    #[test]
    fn multiple_values_for_one_memory_kind_are_preserved() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        add(&c, "mem-1", "mentions", "Alice", "", 1.0).unwrap();
        add(&c, "mem-1", "mentions", "Bob", "", 1.0).unwrap();
        add(&c, "mem-1", "mentions", "Charlie", "", 1.0).unwrap();
        let vals: std::collections::HashSet<String> =
            query_by_memory(&c, "mem-1", Some("mentions"))
                .unwrap()
                .into_iter()
                .map(|a| a.value)
                .collect();
        assert_eq!(
            vals,
            ["Alice", "Bob", "Charlie"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    // PARITY: Mnemosyne tests/test_annotations.py::TestAnnotationStoreMultiValuePreservation::test_add_returns_row_id
    #[test]
    fn add_returns_distinct_row_ids() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        let id1 = add(&c, "mem-1", "mentions", "Alice", "", 1.0).unwrap();
        let id2 = add(&c, "mem-1", "mentions", "Bob", "", 1.0).unwrap();
        assert_ne!(id1, id2);
        assert!(id1 > 0 && id2 > 0);
    }

    // PARITY: Mnemosyne tests/test_annotations.py::TestAnnotationStoreExportImport::test_export_import_round_trip
    // PARITY: Mnemosyne tests/test_annotations.py::TestAnnotationStoreExportImport::test_import_idempotent_on_existing_ids
    #[test]
    fn annotation_export_import_round_trips_and_is_idempotent() {
        let src = Store::open_in_memory().unwrap();
        let dst = Store::open_in_memory().unwrap();
        {
            let c = src.conn.lock().unwrap();
            add(&c, "mem-1", "mentions", "Alice", "extraction", 0.8).unwrap();
            add(&c, "mem-1", "mentions", "Bob", "", 1.0).unwrap();
            add(&c, "mem-2", "fact", "Something interesting here", "", 1.0).unwrap();
        }
        let exported = {
            let c = src.conn.lock().unwrap();
            export_all(&c).unwrap()
        };
        assert_eq!(exported.len(), 3, "export must carry every row");

        let dc = dst.conn.lock().unwrap();
        let (inserted, skipped) = import_all(&dc, &exported).unwrap();
        assert_eq!((inserted, skipped), (3, 0), "fresh import inserts all");
        assert_eq!(
            export_all(&dc).unwrap().len(),
            3,
            "round-trip preserves rows"
        );
        // Re-importing the same export is a no-op (dedup by id).
        let (reins, reskip) = import_all(&dc, &exported).unwrap();
        assert_eq!((reins, reskip), (0, 3), "re-import skips existing ids");
    }

    // PARITY: Mnemosyne tests/test_annotations.py::TestAnnotationStoreQueries::test_query_by_memory_with_kind_filter
    #[test]
    fn query_by_memory_filters_by_kind() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        add(&c, "mem-1", "mentions", "Alice", "", 1.0).unwrap();
        add(&c, "mem-1", "mentions", "Bob", "", 1.0).unwrap();
        add(&c, "mem-1", "fact", "Some fact about mem-1 here", "", 1.0).unwrap();
        assert_eq!(query_by_memory(&c, "mem-1", None).unwrap().len(), 3);
        assert_eq!(
            query_by_memory(&c, "mem-1", Some("mentions"))
                .unwrap()
                .len(),
            2
        );
    }
}
