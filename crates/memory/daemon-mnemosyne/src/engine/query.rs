// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! SQL row-access layer for the BEAM [`Engine`]: scoped single-row fetches over the working and
//! episodic tiers, the per-row recall-stat bump used by the polyphonic pipeline, and the embedding
//! cosine maps. Split out of `engine.rs` (W-MNEMO). The linear recall pipeline owns its own
//! candidate SQL in `engine/recall.rs` (the faithful `beam.py` port).

use super::*;
use crate::config::RecallScope;
use crate::util;
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};
use std::collections::HashMap;

/// The working-tier result projection shared by the fetch paths (`id, content, source,
/// timestamp, importance, veracity, trust_tier`).
const WORKING_SELECT: &str =
    "SELECT id, content, source, timestamp, importance, veracity, trust_tier FROM working_memory";

/// The episodic-tier result projection (the working columns plus the integer `tier` level).
const EPISODIC_SELECT: &str = "SELECT id, content, source, timestamp, importance, veracity, \
     trust_tier, tier FROM episodic_memory";

impl Engine {
    /// Fetch a single row by id from a tier, scope-filtered. Shared by the working/episodic
    /// single-row fetches.
    fn scoped_fetch(
        &self,
        conn: &Connection,
        select_from: &str,
        map: fn(&rusqlite::Row<'_>) -> MemoryRow,
        scope: &RecallScope,
        id: &str,
    ) -> Result<Option<MemoryRow>> {
        let (scope_sql, scope_params) = self.scope_clause(scope);
        let sql = format!(
            "{select_from} \
             WHERE id = ? AND (valid_until IS NULL OR valid_until > ?) \
             AND superseded_by IS NULL{scope_sql}",
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut bind = vec![Value::Text(id.to_string()), Value::Text(util::now_iso())];
        bind.extend(scope_params);
        let mut rows = stmt.query_map(params_from_iter(bind), |r| Ok(map(r)))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Fetch a single live working row by id, scope-filtered.
    pub(crate) fn fetch_working(
        &self,
        conn: &Connection,
        id: &str,
        scope: &RecallScope,
    ) -> Result<Option<MemoryRow>> {
        self.scoped_fetch(conn, WORKING_SELECT, working_row, scope, id)
    }

    /// Fetch a single live episodic row by id, scope-filtered.
    pub(crate) fn fetch_episodic(
        &self,
        conn: &Connection,
        id: &str,
        scope: &RecallScope,
    ) -> Result<Option<MemoryRow>> {
        self.scoped_fetch(conn, EPISODIC_SELECT, episodic_row, scope, id)
    }

    /// Bump `recall_count` / `last_recalled` for the returned rows in their source tier (`beam.py`
    /// L6084-L6119). Used by the polyphonic pipeline; the linear path batches its own scoped bump.
    pub(crate) fn bump_recall(&self, conn: &Connection, rows: &[MemoryRow]) -> Result<()> {
        let now = util::now_iso();
        for row in rows {
            let table = match row.tier {
                Tier::Working => "working_memory",
                Tier::Episodic => "episodic_memory",
                Tier::Memoria | Tier::MemoriaSource | Tier::Fact => continue,
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
        ..Default::default()
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
        ..Default::default()
    }
}
