// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The embedding index store: a standalone SQLite database (separate from `daemon-store`) holding
//! the file/chunk tables and their little-endian `f32` embedding blobs.
//!
//! Concurrency mirrors `daemon-store`: a single `Mutex<Connection>` serializes all access and the
//! database runs in WAL mode. The similarity search is a brute-force cosine scan over every chunk
//! (the design's explicit tradeoff — no ANN index), run inside the caller's `spawn_blocking`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use daemon_core::cosine;
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};

use crate::{IndexError, IndexHit};

/// The current schema version, recorded in `meta` so a future ladder change is detectable.
pub(crate) const SCHEMA_VERSION: &str = "1";

/// The migration ladder. Append `M::up(...)` for each future change — never edit a released
/// migration. Pragmas (WAL) are applied outside the migration transaction, as in `daemon-store`.
static MIGRATIONS: LazyLock<Migrations<'static>> = LazyLock::new(|| {
    Migrations::new(vec![M::up(
        r#"
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE files (
    path         TEXT PRIMARY KEY,
    content_hash BLOB NOT NULL,
    mtime_ms     INTEGER NOT NULL,
    size         INTEGER NOT NULL,
    indexed_ms   INTEGER NOT NULL
);
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX chunks_path ON chunks(path);
"#,
    )])
});

/// A stored file's identity columns, for the reconcile pre-filter.
pub(crate) struct FileRow {
    /// The content hash (sha-256) recorded at index time.
    pub content_hash: Vec<u8>,
    /// The file mtime in unix-millis recorded at index time.
    pub mtime_ms: i64,
    /// The file size in bytes recorded at index time.
    pub size: i64,
}

/// One chunk to persist for a file: its 1-based inclusive line span, text, and embedding vector.
pub(crate) struct ChunkRow<'a> {
    pub start_line: usize,
    pub end_line: usize,
    pub text: &'a str,
    pub embedding: &'a [f32],
}

/// The SQLite-backed embedding index store.
pub(crate) struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Open (creating if absent) the index DB at `path`, applying WAL pragmas then the schema ladder.
    pub(crate) fn open(path: &Path) -> Result<Self, IndexError> {
        let mut conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
        )?;
        MIGRATIONS.to_latest(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an ephemeral in-memory index (tests; the single connection keeps it alive).
    #[cfg(test)]
    pub(crate) fn open_in_memory() -> Result<Self, IndexError> {
        let mut conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        MIGRATIONS.to_latest(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Read a `meta` value.
    pub(crate) fn meta_get(&self, key: &str) -> Result<Option<String>, IndexError> {
        Ok(self
            .lock()
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
                r.get::<_, String>(0)
            })
            .optional()?)
    }

    /// Write (upsert) a `meta` value.
    pub(crate) fn meta_set(&self, key: &str, value: &str) -> Result<(), IndexError> {
        self.lock().execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Drop every file + chunk row (a full rebuild after a workspace/model/dims change).
    pub(crate) fn clear(&self) -> Result<(), IndexError> {
        let conn = self.lock();
        conn.execute("DELETE FROM chunks", [])?;
        conn.execute("DELETE FROM files", [])?;
        Ok(())
    }

    /// The stored identity columns for `path`, if indexed.
    pub(crate) fn file_row(&self, path: &str) -> Result<Option<FileRow>, IndexError> {
        Ok(self
            .lock()
            .query_row(
                "SELECT content_hash, mtime_ms, size FROM files WHERE path = ?1",
                [path],
                |r| {
                    Ok(FileRow {
                        content_hash: r.get(0)?,
                        mtime_ms: r.get(1)?,
                        size: r.get(2)?,
                    })
                },
            )
            .optional()?)
    }

    /// The set of all indexed paths (for the reconcile deletion sweep).
    pub(crate) fn all_paths(&self) -> Result<HashSet<String>, IndexError> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT path FROM files")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut set = HashSet::new();
        for row in rows {
            set.insert(row?);
        }
        Ok(set)
    }

    /// Update only the mtime/size stamp of an already-indexed file whose content hash was unchanged
    /// (a touch that did not alter bytes) — avoids a needless re-embed.
    pub(crate) fn touch_file(
        &self,
        path: &str,
        mtime_ms: i64,
        size: i64,
        indexed_ms: i64,
    ) -> Result<(), IndexError> {
        self.lock().execute(
            "UPDATE files SET mtime_ms = ?2, size = ?3, indexed_ms = ?4 WHERE path = ?1",
            params![path, mtime_ms, size, indexed_ms],
        )?;
        Ok(())
    }

    /// Replace a file's row and all its chunks in one transaction (idempotent upsert).
    pub(crate) fn upsert_file(
        &self,
        path: &str,
        content_hash: &[u8],
        mtime_ms: i64,
        size: i64,
        indexed_ms: i64,
        chunks: &[ChunkRow<'_>],
    ) -> Result<(), IndexError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        // ON DELETE CASCADE clears old chunks when the file row is replaced.
        tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
        tx.execute(
            "INSERT INTO files(path, content_hash, mtime_ms, size, indexed_ms)
             VALUES(?1, ?2, ?3, ?4, ?5)",
            params![path, content_hash, mtime_ms, size, indexed_ms],
        )?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks(path, start_line, end_line, text, embedding)
                 VALUES(?1, ?2, ?3, ?4, ?5)",
            )?;
            for c in chunks {
                stmt.execute(params![
                    path,
                    c.start_line as i64,
                    c.end_line as i64,
                    c.text,
                    encode_embedding(c.embedding),
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Delete a file's row (and its chunks, via cascade).
    pub(crate) fn delete_file(&self, path: &str) -> Result<(), IndexError> {
        self.lock()
            .execute("DELETE FROM files WHERE path = ?1", [path])?;
        Ok(())
    }

    /// Delete every file not present in `keep` (the reconcile removal sweep).
    pub(crate) fn delete_absent(&self, keep: &HashSet<String>) -> Result<(), IndexError> {
        let stale: Vec<String> = self
            .all_paths()?
            .into_iter()
            .filter(|p| !keep.contains(p))
            .collect();
        let conn = self.lock();
        for path in &stale {
            conn.execute("DELETE FROM files WHERE path = ?1", [path])?;
        }
        Ok(())
    }

    /// Brute-force cosine top-`k` over every chunk, filtered to `dir_filters` (root-relative dir
    /// prefixes; empty = whole index). Runs under the connection mutex — call inside `spawn_blocking`.
    pub(crate) fn topk(
        &self,
        query: &[f32],
        k: usize,
        dir_filters: &[String],
    ) -> Result<Vec<IndexHit>, IndexError> {
        let conn = self.lock();
        let mut stmt =
            conn.prepare("SELECT path, start_line, end_line, text, embedding FROM chunks")?;
        let mut scored: Vec<IndexHit> = Vec::new();
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Vec<u8>>(4)?,
            ))
        })?;
        for row in rows {
            let (path, start_line, end_line, text, blob) = row?;
            if !path_in_filters(&path, dir_filters) {
                continue;
            }
            let emb = decode_embedding(&blob);
            let score = cosine(query, &emb);
            scored.push(IndexHit {
                path,
                start_line: start_line as usize,
                end_line: end_line as usize,
                score,
                snippet: text,
            });
        }
        // Highest cosine first; a stable, deterministic tiebreak on (path, start_line).
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.path.cmp(&b.path))
                .then_with(|| a.start_line.cmp(&b.start_line))
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// The current schema `user_version` (test observability).
    #[cfg(test)]
    pub(crate) fn schema_version(&self) -> i64 {
        self.lock()
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap()
    }

    /// The live schema DDL dump (for the golden-drift test).
    #[cfg(test)]
    pub(crate) fn dump_schema(&self) -> String {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name")
            .unwrap();
        let mut out = String::new();
        for sql in stmt.query_map([], |r| r.get::<_, String>(0)).unwrap() {
            out.push_str(sql.unwrap().trim());
            out.push_str(";\n");
        }
        out
    }
}

/// Whether `path` falls under one of the `dir_filters` (root-relative dir prefixes). An empty list
/// (or an empty-string filter) matches everything.
fn path_in_filters(path: &str, dir_filters: &[String]) -> bool {
    if dir_filters.is_empty() {
        return true;
    }
    dir_filters
        .iter()
        .any(|f| f.is_empty() || path == f.as_str() || path.starts_with(&format!("{f}/")))
}

/// Encode an embedding as little-endian `f32` bytes.
fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// Decode a little-endian `f32` blob back into a vector (trailing partial bytes ignored).
fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit_paths(hits: &[IndexHit]) -> Vec<&str> {
        hits.iter().map(|h| h.path.as_str()).collect()
    }

    #[test]
    fn fresh_db_stamps_latest_migration() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.schema_version(), 1);
    }

    #[test]
    fn embedding_blob_roundtrips() {
        let v = vec![1.0f32, -2.5, 3.25, 0.0];
        let bytes = encode_embedding(&v);
        assert_eq!(bytes.len(), 16);
        assert_eq!(decode_embedding(&bytes), v);
    }

    #[test]
    fn upsert_then_topk_orders_by_cosine() {
        let store = Store::open_in_memory().unwrap();
        // Two files, each one chunk; one aligned with the query, one orthogonal.
        store
            .upsert_file(
                "a.rs",
                &[0u8; 32],
                1,
                10,
                1,
                &[ChunkRow {
                    start_line: 1,
                    end_line: 2,
                    text: "aligned",
                    embedding: &[1.0, 0.0, 0.0],
                }],
            )
            .unwrap();
        store
            .upsert_file(
                "b.rs",
                &[0u8; 32],
                1,
                10,
                1,
                &[ChunkRow {
                    start_line: 1,
                    end_line: 2,
                    text: "orthogonal",
                    embedding: &[0.0, 1.0, 0.0],
                }],
            )
            .unwrap();
        let hits = store.topk(&[1.0, 0.0, 0.0], 10, &[]).unwrap();
        assert_eq!(hit_paths(&hits), vec!["a.rs", "b.rs"]);
        assert!(hits[0].score > hits[1].score);
        assert_eq!((hits[0].start_line, hits[0].end_line), (1, 2));

        // k caps the result count.
        assert_eq!(store.topk(&[1.0, 0.0, 0.0], 1, &[]).unwrap().len(), 1);
    }

    #[test]
    fn dir_filters_narrow_by_prefix() {
        let store = Store::open_in_memory().unwrap();
        for path in [
            "src/lib.rs",
            "src/app/main.rs",
            "docs/readme.md",
            "srcfoo.rs",
        ] {
            store
                .upsert_file(
                    path,
                    &[0u8; 32],
                    1,
                    1,
                    1,
                    &[ChunkRow {
                        start_line: 1,
                        end_line: 1,
                        text: path,
                        embedding: &[1.0, 0.0],
                    }],
                )
                .unwrap();
        }
        let hits = store.topk(&[1.0, 0.0], 10, &["src".to_string()]).unwrap();
        let mut paths = hit_paths(&hits);
        paths.sort();
        // `src` matches `src/...` but NOT the sibling `srcfoo.rs` (segment-aware prefix).
        assert_eq!(paths, vec!["src/app/main.rs", "src/lib.rs"]);

        // A deeper filter narrows further.
        let deep = store
            .topk(&[1.0, 0.0], 10, &["src/app".to_string()])
            .unwrap();
        assert_eq!(hit_paths(&deep), vec!["src/app/main.rs"]);

        // The empty-string filter matches everything.
        assert_eq!(
            store
                .topk(&[1.0, 0.0], 10, &["".to_string()])
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn upsert_replaces_prior_chunks_and_delete_cascades() {
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_file(
                "a.rs",
                &[1u8; 32],
                1,
                1,
                1,
                &[
                    ChunkRow {
                        start_line: 1,
                        end_line: 1,
                        text: "one",
                        embedding: &[1.0],
                    },
                    ChunkRow {
                        start_line: 2,
                        end_line: 2,
                        text: "two",
                        embedding: &[1.0],
                    },
                ],
            )
            .unwrap();
        assert_eq!(store.topk(&[1.0], 10, &[]).unwrap().len(), 2);
        // Re-upsert with a single chunk replaces the pair.
        store
            .upsert_file(
                "a.rs",
                &[2u8; 32],
                2,
                2,
                2,
                &[ChunkRow {
                    start_line: 1,
                    end_line: 1,
                    text: "only",
                    embedding: &[1.0],
                }],
            )
            .unwrap();
        assert_eq!(store.topk(&[1.0], 10, &[]).unwrap().len(), 1);
        // Deleting the file cascades its chunks away.
        store.delete_file("a.rs").unwrap();
        assert!(store.topk(&[1.0], 10, &[]).unwrap().is_empty());
    }

    #[test]
    fn delete_absent_removes_only_unseen_paths() {
        let store = Store::open_in_memory().unwrap();
        for path in ["keep.rs", "drop.rs"] {
            store
                .upsert_file(
                    path,
                    &[0u8; 32],
                    1,
                    1,
                    1,
                    &[ChunkRow {
                        start_line: 1,
                        end_line: 1,
                        text: path,
                        embedding: &[1.0],
                    }],
                )
                .unwrap();
        }
        let keep: HashSet<String> = ["keep.rs".to_string()].into_iter().collect();
        store.delete_absent(&keep).unwrap();
        assert_eq!(store.all_paths().unwrap(), keep);
    }

    #[test]
    fn meta_roundtrips() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.meta_get("k").unwrap(), None);
        store.meta_set("k", "v1").unwrap();
        store.meta_set("k", "v2").unwrap();
        assert_eq!(store.meta_get("k").unwrap().as_deref(), Some("v2"));
    }

    /// On-disk schema-drift gate (mirrors daemon-store's `schema_matches_golden`): the live schema
    /// must match the committed golden. Any DDL change must add a migration AND refresh the golden
    /// via `DAEMON_UPDATE_SCHEMA=1 cargo test -p daemon-workspace-index schema_matches_golden`.
    #[test]
    fn schema_matches_golden() {
        let store = Store::open_in_memory().unwrap();
        let dump = store.dump_schema();
        let golden_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/schema.golden.sql");
        if std::env::var_os("DAEMON_UPDATE_SCHEMA").is_some() {
            std::fs::write(golden_path, &dump).expect("write golden");
            return;
        }
        let golden = std::fs::read_to_string(golden_path).unwrap_or_default();
        assert_eq!(
            dump.trim(),
            golden.trim(),
            "schema drift: add a migration (M::up) and refresh src/schema.golden.sql via \
             DAEMON_UPDATE_SCHEMA=1",
        );
    }
}
