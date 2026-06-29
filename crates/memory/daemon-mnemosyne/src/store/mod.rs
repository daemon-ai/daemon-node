// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The SQLite storage layer — port of `beam.py`'s connection + schema setup (L404-L1026).
//!
//! A single `Mutex<Connection>` serializes all access (the workspace convention, see
//! `daemon-store/src/sqlite.rs`), with WAL + `busy_timeout`. One file per bank
//! (`banks.py`). sqlite-vec registration (the `vec-ext` feature) is the only `unsafe` in the crate.

pub mod schema;

use crate::error::Result;
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// The SQLite-backed BEAM store (one file per bank).
pub struct Store {
    /// The serialized connection (WAL allows concurrent readers if a pool is added later).
    pub conn: Mutex<Connection>,
}

impl Store {
    /// Open (creating if absent) a bank database at `path`, applying the full schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Auto-extensions only attach to connections opened *after* registration.
        register_vec_extension();
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Open an ephemeral in-memory store (tests).
    pub fn open_in_memory() -> Result<Self> {
        register_vec_extension();
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        conn.execute_batch(schema::SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

/// Register the sqlite-vec extension as a process auto-extension (idempotent; `vec-ext` only).
///
/// This is the single `unsafe` boundary in the crate (a `RawAutoExtension` transmute, per the
/// sqlite-vec rusqlite guide). With the feature off this is a no-op and the engine uses the f32-BLOB
/// cosine fallback.
#[cfg(feature = "vec-ext")]
fn register_vec_extension() {
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| unsafe {
        use rusqlite::auto_extension::{register_auto_extension, RawAutoExtension};
        let raw: RawAutoExtension =
            std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const () as usize);
        let _ = register_auto_extension(raw);
    });
}

#[cfg(not(feature = "vec-ext"))]
fn register_vec_extension() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_applies_cleanly() {
        let store = Store::open_in_memory().expect("open");
        let conn = store.conn.lock().unwrap();
        // FTS5 must be available in the bundled build; the virtual tables must exist.
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name IN \
                 ('working_memory','episodic_memory','scratchpad','triples','annotations', \
                  'canonical_facts','consolidated_facts','fts_working','fts_episodes')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 9);
    }
}
