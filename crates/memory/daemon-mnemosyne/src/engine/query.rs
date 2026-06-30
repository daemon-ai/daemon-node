// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! SQL row-access layer for the BEAM [`Engine`]: scoped scan/fetch over the working and episodic
//! tiers, FTS5 match execution, binary-vector loading, recall-stat bumps, the result-row mappers,
//! and the embedding cosine maps. Split out of `engine.rs` (W-MNEMO).

use super::*;
use crate::config::RecallScope;
use crate::util;
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};
use std::collections::HashMap;

/// The working-tier result projection shared by the scan/fetch paths (`id, content, source,
/// timestamp, importance, veracity, trust_tier`).
const WORKING_SELECT: &str =
    "SELECT id, content, source, timestamp, importance, veracity, trust_tier FROM working_memory";

/// The episodic-tier result projection (the working columns plus the integer `tier` level).
const EPISODIC_SELECT: &str = "SELECT id, content, source, timestamp, importance, veracity, \
     trust_tier, tier FROM episodic_memory";

/// A tier's scoped-query recipe: its `SELECT ... FROM <table>` projection, the recall scope to
/// filter by, and the row mapper. Bundles the stable params shared by [`Engine::scoped_scan`] /
/// [`Engine::scoped_fetch`] so neither helper carries an excess argument count.
struct TierQuery<'a> {
    select_from: &'static str,
    scope: &'a RecallScope,
    map: fn(&rusqlite::Row<'_>) -> MemoryRow,
}

impl<'a> TierQuery<'a> {
    fn working(scope: &'a RecallScope) -> Self {
        Self {
            select_from: WORKING_SELECT,
            scope,
            map: working_row,
        }
    }

    fn episodic(scope: &'a RecallScope) -> Self {
        Self {
            select_from: EPISODIC_SELECT,
            scope,
            map: episodic_row,
        }
    }
}

impl Engine {
    /// Run an FTS5 `MATCH` query (`sql` selecting `(id, bm25)`), returning `id -> normalized BM25`
    /// for the hits. An empty token list (or a query with no usable terms) yields no hits.
    pub(crate) fn fts_search(
        &self,
        conn: &Connection,
        sql: &str,
        q_tokens: &[String],
        limit: usize,
    ) -> Result<HashMap<String, f64>> {
        let Some(match_str) = fts_match_string(q_tokens) else {
            return Ok(HashMap::new());
        };
        let mut stmt = conn.prepare(sql)?;
        let mut map = HashMap::new();
        let rows = stmt.query_map(params![match_str, limit as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
        })?;
        for row in rows {
            let (id, bm25) = row?;
            map.insert(id, normalize_bm25(bm25));
        }
        Ok(map)
    }

    /// Recency/importance fallback scan over a tier (the candidate floor), filtered by the
    /// multi-agent recall scope. Shared by the working/episodic scans.
    fn scoped_scan(
        &self,
        conn: &Connection,
        spec: &TierQuery,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        let (scope_sql, scope_params) = self.scope_clause(spec.scope);
        let sql = format!(
            "{} \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL{scope_sql} \
             ORDER BY importance DESC, timestamp DESC LIMIT ?",
            spec.select_from,
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut bind = scope_params;
        bind.push(Value::Integer(limit as i64));
        let map = spec.map;
        let rows = stmt
            .query_map(params_from_iter(bind), |r| Ok(map(r)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetch a single row by id (for FTS hits beyond the fallback window), scope-filtered. Shared by
    /// the working/episodic single-row fetches.
    fn scoped_fetch(
        &self,
        conn: &Connection,
        spec: &TierQuery,
        id: &str,
    ) -> Result<Option<MemoryRow>> {
        let (scope_sql, scope_params) = self.scope_clause(spec.scope);
        let sql = format!(
            "{} \
             WHERE id = ? AND (valid_until IS NULL) AND superseded_by IS NULL{scope_sql}",
            spec.select_from,
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut bind = vec![Value::Text(id.to_string())];
        bind.extend(scope_params);
        let map = spec.map;
        let mut rows = stmt.query_map(params_from_iter(bind), |r| Ok(map(r)))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Recency/importance fallback scan over working memory (the candidate floor), scope-filtered.
    pub(crate) fn scan_working(
        &self,
        conn: &Connection,
        limit: usize,
        scope: &RecallScope,
    ) -> Result<Vec<MemoryRow>> {
        self.scoped_scan(conn, &TierQuery::working(scope), limit)
    }

    /// Fetch a single working row by id (for FTS hits beyond the fallback window), scope-filtered.
    pub(crate) fn fetch_working(
        &self,
        conn: &Connection,
        id: &str,
        scope: &RecallScope,
    ) -> Result<Option<MemoryRow>> {
        self.scoped_fetch(conn, &TierQuery::working(scope), id)
    }

    /// Recency/importance fallback scan over episodic memory, scope-filtered.
    pub(crate) fn scan_episodic(
        &self,
        conn: &Connection,
        limit: usize,
        scope: &RecallScope,
    ) -> Result<Vec<MemoryRow>> {
        self.scoped_scan(conn, &TierQuery::episodic(scope), limit)
    }

    /// Fetch a single episodic row by id (for FTS hits beyond the fallback window), scope-filtered.
    pub(crate) fn fetch_episodic(
        &self,
        conn: &Connection,
        id: &str,
        scope: &RecallScope,
    ) -> Result<Option<MemoryRow>> {
        self.scoped_fetch(conn, &TierQuery::episodic(scope), id)
    }

    /// Load the packed MIB `binary_vector` blobs for episodic rows, keyed by memory id.
    pub(crate) fn load_binary_vectors(
        &self,
        conn: &Connection,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let mut stmt = conn.prepare(
            "SELECT id, binary_vector FROM episodic_memory WHERE binary_vector IS NOT NULL",
        )?;
        let mut map = HashMap::new();
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (id, blob) = row?;
            map.insert(id, blob);
        }
        Ok(map)
    }

    /// Bump `recall_count` / `last_recalled` for the returned rows in their source tier (`beam.py`
    /// L6084-L6119).
    pub(crate) fn bump_recall(&self, conn: &Connection, rows: &[MemoryRow]) -> Result<()> {
        let now = util::now_iso();
        for row in rows {
            let table = match row.tier {
                Tier::Working => "working_memory",
                Tier::Episodic => "episodic_memory",
            };
            conn.execute(
                &format!(
                    "UPDATE {table} SET recall_count = recall_count + 1, last_recalled = ?2 \
                     WHERE id = ?1"
                ),
                params![row.id, now],
            )?;
        }
        Ok(())
    }
}

/// Load the stored f32 embeddings (`memory_embeddings.embedding_json`), keyed by memory id.
pub(crate) fn load_embeddings(conn: &Connection) -> Result<HashMap<String, Vec<f32>>> {
    let mut stmt = conn.prepare("SELECT memory_id, embedding_json FROM memory_embeddings")?;
    let mut map = HashMap::new();
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (id, json) = row?;
        if let Ok(vec) = serde_json::from_str::<Vec<f32>>(&json) {
            map.insert(id, vec);
        }
    }
    Ok(map)
}

/// Cosine-similarity map `memory_id -> cos(query, embedding)` over `memory_embeddings`.
///
/// With the `vec-ext` feature this routes through sqlite-vec's native `vec_distance_cosine`
/// scalar (`sim = 1 - distance`) so the vector math runs inside SQLite; otherwise it loads the
/// f32 BLOBs (here JSON arrays) into Rust and uses [`daemon_core::cosine`]. Both paths are
/// behaviour-equivalent (modulo float precision), and the native path silently falls back to the
/// Rust path on any error (e.g. a stray dimension mismatch) so callers always get a full map.
pub(crate) fn cosine_sim_map(conn: &Connection, query: &[f32]) -> Result<HashMap<String, f64>> {
    #[cfg(feature = "vec-ext")]
    {
        match native_cosine_sim_map(conn, query) {
            Ok(map) => return Ok(map),
            Err(e) => {
                tracing::debug!(error = %e, "native vec0 cosine failed; falling back to f32-BLOB");
            }
        }
    }
    let stored = load_embeddings(conn)?;
    Ok(stored
        .iter()
        .map(|(id, v)| (id.clone(), daemon_core::cosine(query, v) as f64))
        .collect())
}

/// Native sqlite-vec cosine path: `1 - vec_distance_cosine(query, embedding)` computed in SQLite.
#[cfg(feature = "vec-ext")]
pub(crate) fn native_cosine_sim_map(
    conn: &Connection,
    query: &[f32],
) -> Result<HashMap<String, f64>> {
    let qjson = serde_json::to_string(query).unwrap_or_else(|_| "[]".to_string());
    let mut stmt = conn.prepare(
        "SELECT memory_id, \
         1.0 - vec_distance_cosine(vec_f32(?1), vec_f32(embedding_json)) \
         FROM memory_embeddings",
    )?;
    let mut map = HashMap::new();
    let rows = stmt.query_map(params![qjson], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
    })?;
    for row in rows {
        let (id, sim) = row?;
        map.insert(id, sim);
    }
    Ok(map)
}

/// Extract the columns common to the working/episodic projections (`id, content, source,
/// timestamp, importance, veracity, trust_tier`, indices `0..=6`), shared by the row mappers.
fn base_cols(r: &rusqlite::Row<'_>) -> (String, String, String, String, f64, String, String) {
    (
        r.get(0).unwrap_or_default(),
        r.get(1).unwrap_or_default(),
        r.get::<_, Option<String>>(2)
            .ok()
            .flatten()
            .unwrap_or_default(),
        r.get::<_, Option<String>>(3)
            .ok()
            .flatten()
            .unwrap_or_default(),
        r.get(4).unwrap_or(0.5),
        r.get::<_, Option<String>>(5)
            .ok()
            .flatten()
            .unwrap_or_default(),
        r.get::<_, Option<String>>(6)
            .ok()
            .flatten()
            .unwrap_or_default(),
    )
}

/// Map a working-memory result row into a [`MemoryRow`] at tier [`Tier::Working`].
fn working_row(r: &rusqlite::Row<'_>) -> MemoryRow {
    let (id, content, source, timestamp, importance, veracity, trust_tier) = base_cols(r);
    MemoryRow {
        id,
        content,
        source,
        timestamp,
        importance,
        veracity,
        trust_tier,
        tier: Tier::Working,
        tier_level: 1,
        score: 0.0,
    }
}

/// Map an episodic result row (working columns + `tier`) into a [`MemoryRow`] at tier
/// [`Tier::Episodic`], carrying the integer tier level for the post-multiplier.
fn episodic_row(r: &rusqlite::Row<'_>) -> MemoryRow {
    let (id, content, source, timestamp, importance, veracity, trust_tier) = base_cols(r);
    MemoryRow {
        id,
        content,
        source,
        timestamp,
        importance,
        veracity,
        trust_tier,
        tier: Tier::Episodic,
        tier_level: r.get::<_, Option<i64>>(7).ok().flatten().unwrap_or(1),
        score: 0.0,
    }
}

/// Build an FTS5 `MATCH` expression from query tokens (`"a" OR "b" OR ...`), quoting each term so
/// punctuation/operators can never break the query. `None` when there are no usable terms.
fn fts_match_string(q_tokens: &[String]) -> Option<String> {
    if q_tokens.is_empty() {
        return None;
    }
    let parts: Vec<String> = q_tokens
        .iter()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    Some(parts.join(" OR "))
}

/// Map SQLite FTS5 `bm25()` (more-negative = better) onto `[0, 1)` (`raw / (1 + raw)`), so a missed
/// row contributes `0` and a strong match approaches `1`.
fn normalize_bm25(bm25: f64) -> f64 {
    let raw = (-bm25).max(0.0);
    raw / (1.0 + raw)
}
