// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Embedding access for the BEAM [`Engine`]: the stored-embedding loader and the optional
//! sqlite-vec native cosine map. The recall pipelines own their candidate SQL in
//! `engine/recall.rs` (the faithful `beam.py` port).

use super::*;
use rusqlite::Connection;
use std::collections::HashMap;

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

/// Native sqlite-vec cosine path: `1 - vec_distance_cosine(query, embedding)` computed in SQLite.
#[cfg(feature = "vec-ext")]
pub(crate) fn native_cosine_sim_map(
    conn: &Connection,
    query: &[f32],
) -> Result<HashMap<String, f64>> {
    use rusqlite::params;
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
