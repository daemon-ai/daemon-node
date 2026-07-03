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
use rusqlite_migration::{Migrations, M};
use std::path::Path;
use std::sync::{LazyLock, Mutex};

/// The schema-migration ladder, gated by `PRAGMA user_version` (rusqlite_migration). `M1` is the
/// full bank schema; future schema changes append an `M::up("…")`. `busy_timeout` is applied in
/// [`Store::init`] before `to_latest`; the `vec-ext` `vec0` virtual tables in the schema resolve
/// because [`register_vec_extension`] runs before the connection is opened.
static MIGRATIONS: LazyLock<Migrations<'static>> =
    LazyLock::new(|| Migrations::new(vec![M::up(schema::SCHEMA), M::up(schema::SCHEMA_V2)]));

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

    fn init(mut conn: Connection) -> Result<Self> {
        // Connection pragmas applied OUTSIDE the (transactional) migration ladder: `journal_mode`
        // cannot change inside a transaction.
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        // Schema is owned by the `PRAGMA user_version` ladder.
        MIGRATIONS.to_latest(&mut conn)?;
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

    /// The migration ladder is internally consistent and a fresh store opens at the latest version.
    #[test]
    fn migration_ladder_valid() {
        assert!(MIGRATIONS.validate().is_ok());
        let store = Store::open_in_memory().expect("open");
        let version: i64 = store
            .conn
            .lock()
            .unwrap()
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 2, "fresh DB is stamped to the latest migration");
    }

    fn dump_schema(conn: &Connection) -> String {
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

    /// On-disk schema-drift gate: the live schema must match the committed golden. Any DDL change
    /// must go through a new migration AND refresh the golden — run `DAEMON_UPDATE_SCHEMA=1 cargo
    /// test -p daemon-mnemosyne schema_matches_golden`.
    #[test]
    fn schema_matches_golden() {
        let store = Store::open_in_memory().expect("open");
        let dump = dump_schema(&store.conn.lock().unwrap());
        let golden_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/store/schema.golden.sql");
        if std::env::var_os("DAEMON_UPDATE_SCHEMA").is_some() {
            std::fs::write(golden_path, &dump).expect("write golden");
            return;
        }
        let golden = std::fs::read_to_string(golden_path).unwrap_or_default();
        assert_eq!(
            dump.trim(),
            golden.trim(),
            "schema drift: add a migration (M::up) and refresh src/store/schema.golden.sql via \
             DAEMON_UPDATE_SCHEMA=1",
        );
    }

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
