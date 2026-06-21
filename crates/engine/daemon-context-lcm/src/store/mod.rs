//! The LCM summary store (skeleton): a single SQLite file per bank holding the summary DAG.
//!
//! Concurrency model mirrors `daemon-mnemosyne`: a serialized [`Mutex<Connection>`] in WAL mode. The
//! deep port may evolve this into the dedicated store-actor thread the spec recommends
//! (`daemon-context-lcm-port-spec.md` §4.4); the seam above it does not change.

pub mod schema;

use crate::error::Result;
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// One node in the summary DAG (a compacted span of conversation).
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryNode {
    /// The node's row id.
    pub node_id: i64,
    /// The session the summary belongs to.
    pub session_id: String,
    /// The DAG depth (0 = a summary of raw turns; higher = a summary of summaries).
    pub depth: i64,
    /// The summary text.
    pub summary: String,
    /// The summary's own token count.
    pub token_count: i64,
    /// The token count of the span it summarized.
    pub source_token_count: i64,
    /// The unix timestamp (seconds) the node was created.
    pub created_at: f64,
}

/// The serialized summary store.
pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Open (or create) the store at `path`, applying the schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Open an in-memory store (tests / ephemeral nodes).
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
        )?;
        conn.execute_batch(schema::SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Record a compaction as a summary node, returning its row id.
    pub fn record_summary(
        &self,
        session_id: &str,
        depth: i64,
        summary: &str,
        token_count: i64,
        source_token_count: i64,
        created_at: f64,
    ) -> Result<i64> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO summary_nodes \
             (session_id, depth, summary, token_count, source_token_count, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![session_id, depth, summary, token_count, source_token_count, created_at],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// The number of summary nodes recorded for `session_id`.
    pub fn summary_count(&self, session_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.query_row(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?1",
            [session_id],
            |row| row.get(0),
        )?;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_counts_summaries() {
        let store = Store::open_in_memory().expect("open in-memory store");
        assert_eq!(store.summary_count("s1").unwrap(), 0);
        store
            .record_summary("s1", 0, "a terse summary", 4, 100, 1.0)
            .expect("record");
        assert_eq!(store.summary_count("s1").unwrap(), 1);
        assert_eq!(store.summary_count("other").unwrap(), 0);
    }
}
