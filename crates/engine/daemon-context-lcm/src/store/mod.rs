//! The LCM store (`daemon-context-lcm-port-spec.md` §4): one SQLite file per bank holding the
//! lossless `messages` transcript, the summary DAG, and the lifecycle frontier.
//!
//! Concurrency follows `daemon-mnemosyne`: a serialized [`Mutex<Connection>`] in WAL mode with
//! `synchronous=FULL` (LCM's lossless contract wants per-commit durability — §4.1). The spec's
//! dedicated store-actor (§4.7) is a later refinement; the seam above it does not change.

pub mod schema;

use crate::error::Result;
use rusqlite::{params, Connection, Row};
use std::path::Path;
use std::sync::Mutex;

/// Whether a node's `source_ids` point at raw `messages.store_id`s (a D0 leaf) or child
/// `summary_nodes.node_id`s (a D>=1 condensation) — §5.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceType {
    /// `source_ids` are `messages.store_id`s.
    Messages,
    /// `source_ids` are child `summary_nodes.node_id`s.
    Nodes,
}

impl SourceType {
    /// The stored discriminant.
    pub fn as_str(self) -> &'static str {
        match self {
            SourceType::Messages => "messages",
            SourceType::Nodes => "nodes",
        }
    }

    /// Parse the stored discriminant (anything other than `nodes` is treated as `messages`).
    pub fn parse(s: &str) -> Self {
        match s {
            "nodes" => SourceType::Nodes,
            _ => SourceType::Messages,
        }
    }
}

/// One node in the summary DAG (a compacted span of conversation) — §5.1.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryNode {
    /// The node's row id.
    pub node_id: i64,
    /// The session the summary belongs to.
    pub session_id: String,
    /// The DAG depth (0 = a summary of raw messages; higher = a summary of summaries).
    pub depth: i64,
    /// The summary text.
    pub summary: String,
    /// The summary's own token count.
    pub token_count: i64,
    /// The token count of the span it summarized.
    pub source_token_count: i64,
    /// The lineage: `store_id`s (when `source_type == Messages`) or child `node_id`s.
    pub source_ids: Vec<i64>,
    /// How to interpret `source_ids`.
    pub source_type: SourceType,
    /// The unix timestamp (seconds) the node was created.
    pub created_at: f64,
    /// The earliest source timestamp covered, if known.
    pub earliest_at: Option<f64>,
    /// The latest source timestamp covered, if known.
    pub latest_at: Option<f64>,
    /// A short "Expand for details about: ..." hint.
    pub expand_hint: String,
}

/// A new summary node to persist (`node_id`/FTS handled by the store).
#[derive(Clone, Debug)]
pub struct NewNode {
    /// The session the summary belongs to.
    pub session_id: String,
    /// The DAG depth.
    pub depth: i64,
    /// The summary text.
    pub summary: String,
    /// The summary's own token count.
    pub token_count: i64,
    /// The token count of the span it summarized.
    pub source_token_count: i64,
    /// The lineage source ids.
    pub source_ids: Vec<i64>,
    /// How to interpret `source_ids`.
    pub source_type: SourceType,
    /// Creation timestamp (unix seconds).
    pub created_at: f64,
    /// Earliest source timestamp.
    pub earliest_at: Option<f64>,
    /// Latest source timestamp.
    pub latest_at: Option<f64>,
    /// The expand hint.
    pub expand_hint: String,
}

/// A persisted message row (the lossless transcript identity referenced by D0 nodes) — §4.2.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageRow {
    /// The stable lossless-recovery id.
    pub store_id: i64,
    /// The owning session.
    pub session_id: String,
    /// The platform/source (`""`/NULL normalized to `unknown` on read).
    pub source: String,
    /// `user` | `assistant` | `tool` | `system`.
    pub role: String,
    /// The (FTS-indexed) content.
    pub content: Option<String>,
    /// For a `tool` row: the call id this result answers.
    pub tool_call_id: Option<String>,
    /// For an `assistant` row that issued calls: the tool-calls JSON blob.
    pub tool_calls: Option<String>,
    /// For a `tool` row: the tool name.
    pub tool_name: Option<String>,
    /// The unix timestamp (seconds).
    pub timestamp: f64,
    /// The cached token estimate.
    pub token_estimate: i64,
}

/// A message to append (`store_id`/`timestamp` assigned by the store).
#[derive(Clone, Debug, Default)]
pub struct NewMessage {
    /// The platform/source.
    pub source: String,
    /// The role.
    pub role: String,
    /// The content (FTS-indexed).
    pub content: Option<String>,
    /// For a `tool` row: the call id this result answers.
    pub tool_call_id: Option<String>,
    /// For an `assistant` row: the tool-calls JSON blob.
    pub tool_calls: Option<String>,
    /// For a `tool` row: the tool name.
    pub tool_name: Option<String>,
    /// The token estimate.
    pub token_estimate: i64,
}

/// The serialized LCM store.
pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Open (or create) the store at `path`, applying the v4 schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Self::init(Connection::open(path)?)
    }

    /// Open an in-memory store (tests / ephemeral nodes).
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        // §4.1 pragmas — WAL + FULL durability (LCM is lossless), generous lock wait, bounded WAL.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA busy_timeout=30000;
             PRAGMA wal_autocheckpoint=500;
             PRAGMA journal_size_limit=67108864;
             PRAGMA mmap_size=268435456;",
        )?;
        conn.execute_batch(schema::SCHEMA)?;
        // Record the schema version + a migration marker so a future migration has the same hook.
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?1)",
            params![schema::SCHEMA_VERSION.to_string()],
        )?;
        conn.execute(
            "INSERT OR IGNORE INTO lcm_migration_state(step_name, completed_at) \
             VALUES ('v4_greenfield', strftime('%s','now'))",
            [],
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ---- MessageStore (§4.2) ---------------------------------------------------------------

    /// Append a batch of messages for `session_id` at `timestamp`, returning their assigned
    /// `store_id`s in order. Source blank/empty is normalized to `unknown` on write.
    pub fn append_batch(
        &self,
        session_id: &str,
        msgs: &[NewMessage],
        timestamp: f64,
    ) -> Result<Vec<i64>> {
        let mut conn = self.conn.lock().expect("lcm store poisoned");
        let tx = conn.transaction()?;
        let mut ids = Vec::with_capacity(msgs.len());
        {
            let mut stmt = tx.prepare(
                "INSERT INTO messages \
                 (session_id, source, role, content, tool_call_id, tool_calls, tool_name, \
                  timestamp, token_estimate) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            for m in msgs {
                let source = if m.source.is_empty() {
                    "unknown"
                } else {
                    m.source.as_str()
                };
                stmt.execute(params![
                    session_id,
                    source,
                    m.role,
                    m.content,
                    m.tool_call_id,
                    m.tool_calls,
                    m.tool_name,
                    timestamp,
                    m.token_estimate,
                ])?;
                ids.push(tx.last_insert_rowid());
            }
        }
        tx.commit()?;
        Ok(ids)
    }

    /// Fetch a single message by `store_id`.
    pub fn get_message(&self, store_id: i64) -> Result<Option<MessageRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages WHERE store_id = ?1",
        )?;
        let row = stmt
            .query_row([store_id], map_message)
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        Ok(row)
    }

    /// All messages for `session_id`, oldest first.
    pub fn session_messages(&self, session_id: &str) -> Result<Vec<MessageRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages \
             WHERE session_id = ?1 ORDER BY store_id ASC",
        )?;
        let rows = stmt
            .query_map([session_id], map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The number of messages stored for `session_id`.
    pub fn message_count(&self, session_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// Full-text search the message transcript; returns matching `store_id`s (most relevant first).
    pub fn search_messages(&self, session_id: &str, query: &str, limit: i64) -> Result<Vec<i64>> {
        let sanitized = fts_sanitize(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT m.store_id FROM messages_fts f \
             JOIN messages m ON m.store_id = f.rowid \
             WHERE f.content MATCH ?1 AND m.session_id = ?2 \
             ORDER BY rank LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![sanitized, session_id, limit], |r| r.get::<_, i64>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ---- SummaryDAG (§5) -------------------------------------------------------------------

    /// Persist a summary node, returning its `node_id`.
    pub fn add_node(&self, node: &NewNode) -> Result<i64> {
        let source_ids = serde_json::to_string(&node.source_ids)?;
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "INSERT INTO summary_nodes \
             (session_id, depth, summary, token_count, source_token_count, source_ids, \
              source_type, created_at, earliest_at, latest_at, expand_hint) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                node.session_id,
                node.depth,
                node.summary,
                node.token_count,
                node.source_token_count,
                source_ids,
                node.source_type.as_str(),
                node.created_at,
                node.earliest_at,
                node.latest_at,
                node.expand_hint,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Fetch a node by id.
    pub fn get_node(&self, node_id: i64) -> Result<Option<SummaryNode>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&format!("{NODE_SELECT} WHERE node_id = ?1"))?;
        let row = stmt
            .query_row([node_id], map_node)
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        Ok(row)
    }

    /// All nodes for a session (optionally at one depth), oldest first.
    pub fn get_session_nodes(
        &self,
        session_id: &str,
        depth: Option<i64>,
        limit: i64,
    ) -> Result<Vec<SummaryNode>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        match depth {
            Some(d) => {
                let mut stmt = conn.prepare(&format!(
                    "{NODE_SELECT} WHERE session_id = ?1 AND depth = ?2 \
                     ORDER BY created_at ASC LIMIT ?3"
                ))?;
                let rows = stmt
                    .query_map(params![session_id, d, limit], map_node)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            }
            None => {
                let mut stmt = conn.prepare(&format!(
                    "{NODE_SELECT} WHERE session_id = ?1 ORDER BY created_at ASC LIMIT ?2"
                ))?;
                let rows = stmt
                    .query_map(params![session_id, limit], map_node)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            }
        }
    }

    /// The number of nodes at `depth` for `session_id`.
    pub fn count_at_depth(&self, session_id: &str, depth: i64) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?1 AND depth = ?2",
            params![session_id, depth],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// The condensation feeder (§5.4): nodes at `depth` not yet referenced by any node at
    /// `depth + 1` (i.e. not yet condensed), oldest first.
    pub fn get_uncondensed_at_depth(
        &self,
        session_id: &str,
        depth: i64,
        limit: i64,
    ) -> Result<Vec<SummaryNode>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&format!(
            "{NODE_SELECT} WHERE session_id = ?1 AND depth = ?2 AND node_id NOT IN (\
                 SELECT CAST(j.value AS INTEGER) FROM summary_nodes p, json_each(p.source_ids) j \
                 WHERE p.session_id = ?1 AND p.source_type = 'nodes' AND p.depth = ?2 + 1\
             ) ORDER BY created_at ASC LIMIT ?3"
        ))?;
        let rows = stmt
            .query_map(params![session_id, depth, limit], map_node)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The DAG frontier for assembly (§6.7): every node not referenced by a higher-depth
    /// condensation, highest depth first then oldest.
    pub fn get_uncondensed_frontier(&self, session_id: &str) -> Result<Vec<SummaryNode>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&format!(
            "{NODE_SELECT} WHERE session_id = ?1 AND node_id NOT IN (\
                 SELECT CAST(j.value AS INTEGER) FROM summary_nodes p, json_each(p.source_ids) j \
                 WHERE p.session_id = ?1 AND p.source_type = 'nodes'\
             ) ORDER BY depth DESC, created_at ASC"
        ))?;
        let rows = stmt
            .query_map([session_id], map_node)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The number of summary nodes recorded for `session_id`.
    pub fn summary_count(&self, session_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    // ---- LifecycleStateStore (§4.5) --------------------------------------------------------

    /// Bind the active session for a conversation (idempotent).
    pub fn bind_session(&self, conversation_id: &str, session_id: &str, now: f64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "INSERT INTO lcm_lifecycle_state (conversation_id, current_session_id, \
                 current_bound_at, updated_at) VALUES (?1, ?2, ?3, ?3) \
             ON CONFLICT(conversation_id) DO UPDATE SET \
                 current_session_id = excluded.current_session_id, \
                 current_bound_at = excluded.current_bound_at, updated_at = excluded.updated_at",
            params![conversation_id, session_id, now],
        )?;
        Ok(())
    }

    /// Advance the compaction frontier high-water mark (monotonic — never moves backward).
    pub fn advance_frontier(&self, conversation_id: &str, store_id: i64, now: f64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "INSERT INTO lcm_lifecycle_state (conversation_id, current_frontier_store_id, \
                 updated_at) VALUES (?1, ?2, ?3) \
             ON CONFLICT(conversation_id) DO UPDATE SET \
                 current_frontier_store_id = MAX(current_frontier_store_id, excluded.current_frontier_store_id), \
                 updated_at = excluded.updated_at",
            params![conversation_id, store_id, now],
        )?;
        Ok(())
    }

    /// The current compaction frontier for a conversation (0 if unbound).
    pub fn get_frontier(&self, conversation_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let v = conn
            .query_row(
                "SELECT current_frontier_store_id FROM lcm_lifecycle_state \
                 WHERE conversation_id = ?1",
                [conversation_id],
                |r| r.get::<_, i64>(0),
            )
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(0),
                other => Err(other),
            })?;
        Ok(v)
    }

    /// Record deferred compaction debt (kind + size estimate) for a conversation.
    pub fn record_debt(
        &self,
        conversation_id: &str,
        kind: &str,
        size_estimate: i64,
        now: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "INSERT INTO lcm_lifecycle_state (conversation_id, debt_kind, debt_size_estimate, \
                 debt_updated_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?4) \
             ON CONFLICT(conversation_id) DO UPDATE SET \
                 debt_kind = excluded.debt_kind, debt_size_estimate = excluded.debt_size_estimate, \
                 debt_updated_at = excluded.debt_updated_at, updated_at = excluded.updated_at",
            params![conversation_id, kind, size_estimate, now],
        )?;
        Ok(())
    }
}

/// Column list shared by every `summary_nodes` SELECT (keeps `map_node` indices stable).
const NODE_SELECT: &str = "SELECT node_id, session_id, depth, summary, token_count, \
    source_token_count, source_ids, source_type, created_at, earliest_at, latest_at, expand_hint \
    FROM summary_nodes";

fn map_node(row: &Row<'_>) -> rusqlite::Result<SummaryNode> {
    let source_ids_json: String = row.get(6)?;
    let source_type: String = row.get(7)?;
    Ok(SummaryNode {
        node_id: row.get(0)?,
        session_id: row.get(1)?,
        depth: row.get(2)?,
        summary: row.get(3)?,
        token_count: row.get(4)?,
        source_token_count: row.get(5)?,
        source_ids: serde_json::from_str(&source_ids_json).unwrap_or_default(),
        source_type: SourceType::parse(&source_type),
        created_at: row.get(8)?,
        earliest_at: row.get(9)?,
        latest_at: row.get(10)?,
        expand_hint: row.get(11)?,
    })
}

fn map_message(row: &Row<'_>) -> rusqlite::Result<MessageRow> {
    Ok(MessageRow {
        store_id: row.get(0)?,
        session_id: row.get(1)?,
        source: row.get(2)?,
        role: row.get(3)?,
        content: row.get(4)?,
        tool_call_id: row.get(5)?,
        tool_calls: row.get(6)?,
        tool_name: row.get(7)?,
        timestamp: row.get(8)?,
        token_estimate: row.get(9)?,
    })
}

/// Minimal FTS5 query sanitizer (§11.1): keep alphanumerics/CJK as terms, drop FTS operators by
/// quoting each token, so arbitrary user/search text can't trip a syntax error.
fn fts_sanitize(query: &str) -> String {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{t}\""))
        .collect();
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_counts_summaries() {
        let store = Store::open_in_memory().expect("open in-memory store");
        assert_eq!(store.summary_count("s1").unwrap(), 0);
        store
            .add_node(&NewNode {
                session_id: "s1".into(),
                depth: 0,
                summary: "a terse summary".into(),
                token_count: 4,
                source_token_count: 100,
                source_ids: vec![1, 2, 3],
                source_type: SourceType::Messages,
                created_at: 1.0,
                earliest_at: Some(0.5),
                latest_at: Some(1.0),
                expand_hint: String::new(),
            })
            .expect("record");
        assert_eq!(store.summary_count("s1").unwrap(), 1);
        assert_eq!(store.summary_count("other").unwrap(), 0);
    }

    #[test]
    fn message_round_trip_and_fts_sync() {
        let store = Store::open_in_memory().expect("open");
        let ids = store
            .append_batch(
                "s1",
                &[
                    NewMessage {
                        source: String::new(),
                        role: "user".into(),
                        content: Some("the quick brown fox jumps".into()),
                        token_estimate: 5,
                        ..Default::default()
                    },
                    NewMessage {
                        role: "assistant".into(),
                        content: Some("a lazy dog sleeps".into()),
                        token_estimate: 4,
                        ..Default::default()
                    },
                ],
                10.0,
            )
            .expect("append");
        assert_eq!(ids.len(), 2);
        assert_eq!(store.message_count("s1").unwrap(), 2);

        // Round-trip a row; blank source normalized to "unknown".
        let row = store.get_message(ids[0]).unwrap().unwrap();
        assert_eq!(row.source, "unknown");
        assert_eq!(row.content.as_deref(), Some("the quick brown fox jumps"));

        // FTS triggers kept the index in sync.
        let hits = store.search_messages("s1", "brown fox", 10).unwrap();
        assert_eq!(hits, vec![ids[0]]);
        let none = store.search_messages("s1", "elephant", 10).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn uncondensed_feeder_and_frontier() {
        let store = Store::open_in_memory().expect("open");
        // Four D0 leaves.
        let mut leaves = Vec::new();
        for i in 0..4 {
            let id = store
                .add_node(&NewNode {
                    session_id: "s1".into(),
                    depth: 0,
                    summary: format!("leaf {i}"),
                    token_count: 2,
                    source_token_count: 50,
                    source_ids: vec![i as i64],
                    source_type: SourceType::Messages,
                    created_at: i as f64,
                    earliest_at: None,
                    latest_at: None,
                    expand_hint: String::new(),
                })
                .unwrap();
            leaves.push(id);
        }
        assert_eq!(store.count_at_depth("s1", 0).unwrap(), 4);
        assert_eq!(store.get_uncondensed_at_depth("s1", 0, 10).unwrap().len(), 4);

        // Condense the four leaves into a D1 node.
        store
            .add_node(&NewNode {
                session_id: "s1".into(),
                depth: 1,
                summary: "arc".into(),
                token_count: 2,
                source_token_count: 8,
                source_ids: leaves.clone(),
                source_type: SourceType::Nodes,
                created_at: 5.0,
                earliest_at: None,
                latest_at: None,
                expand_hint: String::new(),
            })
            .unwrap();

        // The leaves are now condensed; only the D1 node is uncondensed.
        assert!(store.get_uncondensed_at_depth("s1", 0, 10).unwrap().is_empty());
        let frontier = store.get_uncondensed_frontier("s1").unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].depth, 1);
        assert_eq!(frontier[0].source_type, SourceType::Nodes);
    }

    #[test]
    fn frontier_is_monotonic() {
        let store = Store::open_in_memory().expect("open");
        store.bind_session("conv", "s1", 1.0).unwrap();
        assert_eq!(store.get_frontier("conv").unwrap(), 0);
        store.advance_frontier("conv", 10, 2.0).unwrap();
        assert_eq!(store.get_frontier("conv").unwrap(), 10);
        // A lower store_id must not move the frontier backward.
        store.advance_frontier("conv", 5, 3.0).unwrap();
        assert_eq!(store.get_frontier("conv").unwrap(), 10);
        store.advance_frontier("conv", 20, 4.0).unwrap();
        assert_eq!(store.get_frontier("conv").unwrap(), 20);
    }
}
