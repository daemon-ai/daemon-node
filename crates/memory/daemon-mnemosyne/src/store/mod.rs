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
static MIGRATIONS: LazyLock<Migrations<'static>> = LazyLock::new(|| {
    Migrations::new(vec![
        M::up(schema::SCHEMA),
        M::up(schema::SCHEMA_V2),
        M::up(schema::SCHEMA_V3),
    ])
});

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
        // Legacy Python banks (`user_version = 0`, old-shape tables already present): reconcile
        // missing columns BEFORE the ladder — its `IF NOT EXISTS` DDL no-ops on existing tables,
        // and its partial indexes reference columns an old bank may not have yet.
        legacy::reconcile_columns(&conn)?;
        // Schema is owned by the `PRAGMA user_version` ladder.
        MIGRATIONS.to_latest(&mut conn)?;
        legacy::e6_backfill(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

/// Legacy Python-bank compatibility — the Rust twin of `beam.py`'s in-place migration ladder.
///
/// Python evolves an existing database by sprinkling `_add_column_if_missing` / `ALTER TABLE`
/// calls through `init_beam` (beam.py L529-L607, L872-L883, L1147-L1154) and auto-runs the E6
/// triplestore split on init (`_ensure_e6_schema_with_migration` L2688,
/// `migrations/e6_triplestore_split.py`). The Rust schema instead folds every column into the
/// `CREATE TABLE` DDL — correct for fresh banks, but a bank created by Python has old-shape
/// tables that `IF NOT EXISTS` leaves untouched, and every Rust statement referencing a newer
/// column would fail. These passes run on every open and are idempotent no-ops on Rust-created
/// banks.
mod legacy {
    use crate::error::Result;
    use rusqlite::Connection;

    /// One expected column, described by the reference (fresh) schema.
    struct ColSpec {
        name: String,
        decl_type: String,
        dflt: Option<String>,
    }

    /// Diff every reference table against the live database and `ALTER TABLE ADD COLUMN` any
    /// column a legacy bank is missing (the `_add_column_if_missing` ladder, generalized: the
    /// expected shape is read from a fresh in-memory database built by the same migration
    /// ladder, so new columns never need a hand-written compat entry). Returns the number of
    /// columns added.
    ///
    /// E3 semantics are preserved (beam.py L578-L607): when `working_memory.consolidated_at`
    /// itself had to be added, every pre-existing row is backfilled as already-consolidated —
    /// pre-E3 Python `sleep()` deleted consolidated rows, so an un-backfilled legacy backlog
    /// would be re-summarized wholesale on the first sleep.
    pub(super) fn reconcile_columns(conn: &Connection) -> Result<usize> {
        let mut reference = Connection::open_in_memory()?;
        super::MIGRATIONS
            .to_latest(&mut reference)
            .map_err(|e| crate::error::Error::Invalid(format!("reference schema: {e}")))?;

        // Real tables only: FTS virtual tables + their shadows (all `fts_`-prefixed here) are
        // owned by the FTS5 module, not column reconciliation.
        let tables: Vec<String> = {
            let mut stmt = reference.prepare(
                "SELECT name FROM sqlite_master WHERE type = 'table' \
                 AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'fts_%' ORDER BY name",
            )?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut added = 0usize;
        for table in &tables {
            // Only reconcile tables that already exist: anything absent is created in full
            // (current shape) by the migration ladder that runs right after this pass.
            let exists: bool = conn
                .query_row(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                    [table],
                    |_| Ok(()),
                )
                .is_ok();
            if !exists {
                continue;
            }
            let expected = table_columns(&reference, table)?;
            let live: std::collections::HashSet<String> = table_columns(conn, table)?
                .into_iter()
                .map(|c| c.name)
                .collect();
            let mut backfill_consolidated_at = false;
            for col in expected {
                if live.contains(&col.name) {
                    continue;
                }
                // `ALTER TABLE ADD COLUMN` rejects non-constant defaults (CURRENT_TIMESTAMP);
                // those columns are added defaultless (NULL on legacy rows), exactly what
                // Python's except-and-continue ladder produced.
                let mut ddl = format!(
                    "ALTER TABLE {table} ADD COLUMN {} {}",
                    col.name, col.decl_type
                );
                if let Some(d) = &col.dflt {
                    if !d.eq_ignore_ascii_case("CURRENT_TIMESTAMP") {
                        ddl.push_str(&format!(" DEFAULT {d}"));
                    }
                }
                conn.execute(&ddl, [])?;
                added += 1;
                if table == "working_memory" && col.name == "consolidated_at" {
                    backfill_consolidated_at = true;
                }
            }
            if backfill_consolidated_at {
                conn.execute(
                    "UPDATE working_memory SET consolidated_at = ?1 WHERE consolidated_at IS NULL",
                    [crate::util::now_iso()],
                )?;
            }
        }
        Ok(added)
    }

    fn table_columns(conn: &Connection, table: &str) -> Result<Vec<ColSpec>> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let rows = stmt.query_map([], |r| {
            Ok(ColSpec {
                name: r.get(1)?,
                decl_type: r.get(2)?,
                dflt: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// The E6 triplestore-split backfill (`migrations/e6_triplestore_split.py`): legacy `triples`
    /// rows whose predicate is an annotation kind are *copied* (never deleted — reversible, like
    /// Python) into `annotations` under the `(memory_id=subject, kind=predicate, value=object)`
    /// mapping. `INSERT OR IGNORE` against the `(memory_id, kind, value)` unique index is the
    /// anti-join: re-running is a no-op. Returns the number of rows copied.
    pub(super) fn e6_backfill(conn: &Connection) -> Result<usize> {
        let kinds = crate::knowledge::annotations::ANNOTATION_KINDS
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(",");
        let n = conn.execute(
            &format!(
                "INSERT OR IGNORE INTO annotations \
                     (memory_id, kind, value, source, confidence, created_at) \
                 SELECT subject, predicate, object, source, COALESCE(confidence, 1.0), created_at \
                 FROM triples WHERE predicate IN ({kinds})"
            ),
            [],
        )?;
        Ok(n)
    }
}

/// Register the sqlite-vec extension as a process auto-extension (idempotent; `vec-ext` only).
///
/// This is the single `unsafe` boundary in the crate (a `RawAutoExtension` transmute, per the
/// sqlite-vec rusqlite guide). With the feature off this is a no-op and the engine uses the f32-BLOB
/// cosine fallback. `pub(crate)` so [`crate::dr`] restore connections resolve vec0 DDL too.
#[cfg(feature = "vec-ext")]
pub(crate) fn register_vec_extension() {
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
pub(crate) fn register_vec_extension() {}

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
        assert_eq!(version, 3, "fresh DB is stamped to the latest migration");
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

    /// Opening a bank created by legacy Python Mnemosyne (old-shape tables, `user_version=0`,
    /// pre-E6 triples, no annotations) must reconcile it in place: missing columns added, the
    /// pre-existing working rows backfilled as consolidated (E3), and annotation-flavored triples
    /// copied (not moved) into `annotations` (E6). Idempotent across reopens.
    #[test]
    fn legacy_python_bank_reconciles_on_open() {
        let dir = std::env::temp_dir().join(format!("mnemosyne-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bank.db");

        {
            // A minimal pre-E3/pre-E6 Python bank: v1 working_memory, Python-shape triples with
            // annotation-flavored and knowledge rows, no annotations table.
            let legacy = Connection::open(&path).unwrap();
            legacy
                .execute_batch(
                    "CREATE TABLE working_memory (
                         id TEXT PRIMARY KEY,
                         content TEXT NOT NULL,
                         source TEXT,
                         timestamp TEXT,
                         session_id TEXT DEFAULT 'default',
                         importance REAL DEFAULT 0.5,
                         metadata_json TEXT,
                         veracity TEXT DEFAULT 'unknown',
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     );
                     INSERT INTO working_memory (id, content, timestamp)
                         VALUES ('legacy1', 'an old python row', '2024-01-01T00:00:00');
                     CREATE TABLE triples (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         subject TEXT NOT NULL,
                         predicate TEXT NOT NULL,
                         object TEXT NOT NULL,
                         valid_from TEXT NOT NULL,
                         valid_until TEXT,
                         source TEXT,
                         confidence REAL DEFAULT 1.0,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     );
                     INSERT INTO triples (subject, predicate, object, valid_from, source)
                         VALUES ('mem1', 'mentions', 'Acme', '2024-01-01', 'regex');
                     INSERT INTO triples (subject, predicate, object, valid_from, source)
                         VALUES ('Maya', 'works_at', 'Acme', '2024-01-01', 'stated');",
                )
                .unwrap();
        }

        let store = Store::open(&path).expect("legacy bank must open");
        {
            let conn = store.conn.lock().unwrap();
            // Column reconcile: newer columns exist and are usable.
            let (consolidated_at, trust_tier, scope): (Option<String>, String, String) = conn
                .query_row(
                    "SELECT consolidated_at, COALESCE(trust_tier, 'STATED'), \
                            COALESCE(scope, 'global') \
                     FROM working_memory WHERE id = 'legacy1'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            // E3 backfill: the pre-existing row is treated as already consolidated.
            assert!(
                consolidated_at.is_some(),
                "legacy rows must be backfilled as consolidated"
            );
            assert_eq!(trust_tier, "STATED");
            assert_eq!(scope, "global");
            // E6 backfill: the annotation-flavored triple was copied, the knowledge one was not,
            // and the source triples row survives (reversible migration).
            let anns: Vec<(String, String, String)> = {
                let mut stmt = conn
                    .prepare("SELECT memory_id, kind, value FROM annotations ORDER BY id")
                    .unwrap();
                let rows = stmt
                    .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                    .unwrap();
                rows.collect::<std::result::Result<Vec<_>, _>>().unwrap()
            };
            assert_eq!(
                anns,
                vec![(
                    "mem1".to_string(),
                    "mentions".to_string(),
                    "Acme".to_string()
                )]
            );
            let triples: i64 = conn
                .query_row("SELECT COUNT(*) FROM triples", [], |r| r.get(0))
                .unwrap();
            assert_eq!(triples, 2, "E6 copies, never deletes");
        }
        drop(store);

        // Reopen: both passes are idempotent.
        let store = Store::open(&path).expect("reopen");
        let conn = store.conn.lock().unwrap();
        let anns: i64 = conn
            .query_row("SELECT COUNT(*) FROM annotations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(anns, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: test_consolidate_fact_concurrency.py::TestReviewHardening::test_consolidator_sets_wal_and_busy_timeout (tests/test_consolidate_fact_concurrency.py:399)
    // parity: test_consolidate_fact_sibling_races.py::TestReviewHardening::test_consolidator_sets_wal_and_busy_timeout (tests/test_consolidate_fact_sibling_races.py:342)
    #[test]
    fn store_sets_wal_and_busy_timeout() {
        // Both pragmas are the contention contract for BEGIN IMMEDIATE writers: without WAL the
        // write lock blocks readers too; without busy_timeout a second writer fails instantly.
        let dir = std::env::temp_dir().join(format!("mnemosyne-pragma-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open(dir.join("pragma_check.db")).expect("open");
        let conn = store.conn.lock().unwrap();
        let mode: String = conn
            .pragma_query_value(None, "journal_mode", |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal", "journal_mode must be WAL");
        let timeout: i64 = conn
            .pragma_query_value(None, "busy_timeout", |r| r.get(0))
            .unwrap();
        assert!(timeout > 0, "busy_timeout must be set, got {timeout}");
        drop(conn);
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
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
