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

/// Minimum fact length kept by the read-time filter (`annotations.py` L89).
pub const MIN_FACT_LENGTH: usize = 10;

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
/// `(memory_id, kind, value)` unique index dedups repeats. Returns the new row id (0 if ignored).
pub fn add(
    conn: &Connection,
    memory_id: &str,
    kind: &str,
    value: &str,
    source: &str,
    confidence: f64,
) -> Result<i64> {
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
    fn mentions_noise_filtered_on_read() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();
        add(&c, "m1", "mentions", "Acme", "regex", 1.0).unwrap();
        add(&c, "m1", "mentions", "System", "regex", 1.0).unwrap();
        let kept = query_by_kind(&c, "mentions", None, true).unwrap();
        assert!(kept.iter().any(|a| a.value == "Acme"));
        assert!(!kept.iter().any(|a| a.value == "System"));
    }
}
