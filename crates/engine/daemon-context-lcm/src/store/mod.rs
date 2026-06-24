//! The LCM store (`daemon-context-lcm-port-spec.md` §4): one SQLite file per bank holding the
//! lossless `messages` transcript, the summary DAG, and the lifecycle frontier.
//!
//! Concurrency follows `daemon-mnemosyne`: a serialized [`Mutex<Connection>`] in WAL mode with
//! `synchronous=FULL` (LCM's lossless contract wants per-commit durability — §4.1). The spec's
//! dedicated store-actor (§4.7) is a later refinement; the seam above it does not change.

pub mod schema;

use crate::error::Result;
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection, Row};
use std::collections::HashSet;
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

/// Optional scoping/filter for a transcript search (§11). Defaults to "everything in the bank".
#[derive(Clone, Debug, Default)]
pub struct MessageFilter<'a> {
    /// Restrict to one session (`None` = search every session in the bank).
    pub session: Option<&'a str>,
    /// Restrict to one role (`user` | `assistant` | `tool` | `system`).
    pub role: Option<&'a str>,
    /// Restrict to one platform/source.
    pub source: Option<&'a str>,
    /// Inclusive lower bound on `timestamp` (unix seconds).
    pub time_from: Option<f64>,
    /// Inclusive upper bound on `timestamp` (unix seconds).
    pub time_to: Option<f64>,
}

/// A message search candidate: the row plus its FTS5 `rank` (lower = better; LIKE candidates use a
/// synthesized rank from term-hit count).
#[derive(Clone, Debug)]
pub struct MessageHit {
    /// The matched message row.
    pub row: MessageRow,
    /// The FTS5 rank (lower is a better match).
    pub rank: f64,
}

/// A node search candidate: the summary node plus its FTS5 `rank` (lower = better).
#[derive(Clone, Debug)]
pub struct NodeHit {
    /// The matched summary node.
    pub node: SummaryNode,
    /// The FTS5 rank (lower is a better match).
    pub rank: f64,
}

/// Bank-wide row counts (base tables + their FTS shadows) for diagnostics (§10.6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreCounts {
    /// Rows in `messages`.
    pub messages: i64,
    /// Rows in the `messages_fts` shadow (should equal `messages`).
    pub messages_fts: i64,
    /// Rows in `summary_nodes`.
    pub nodes: i64,
    /// Rows in the `nodes_fts` shadow (should equal `nodes`).
    pub nodes_fts: i64,
}

/// Presence of the schema's core objects (`lcm_doctor`'s `schema_core_tables` check, §10.6) — the
/// Rust analog of `inspect_lcm_schema_health` (`LCM:db_bootstrap.py:227`).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SchemaHealth {
    /// Core tables/indexes found in `sqlite_master`.
    pub present: Vec<String>,
    /// Expected core tables/indexes that are missing (any => the check fails).
    pub missing: Vec<String>,
    /// The recorded `metadata.schema_version`, if present.
    pub schema_version: Option<i64>,
}

/// SQLite storage posture (`lcm_doctor`'s `sqlite_storage` check) — journal mode, `quick_check`,
/// the backing file path, and the on-disk size.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StoragePosture {
    /// The main database file path (`""` for an in-memory bank).
    pub database_path: String,
    /// Whether the bank is in-memory (no backing file).
    pub in_memory: bool,
    /// `PRAGMA journal_mode` (WAL for a durable bank).
    pub journal_mode: String,
    /// `PRAGMA quick_check` first row (`"ok"` when healthy; drives pass/fail).
    pub quick_check: String,
    /// `page_count * page_size` — the logical database size.
    pub database_size_bytes: i64,
}

/// One node flagged by the summary-quality diagnostic (worst compression ratios first).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct WorstNode {
    /// The node row id.
    pub node_id: i64,
    /// The session the node belongs to.
    pub session_id: String,
    /// The DAG depth.
    pub depth: i64,
    /// The summarized span's token count.
    pub source_token_count: i64,
    /// The summary's own token count.
    pub token_count: i64,
    /// `source_token_count / token_count` (rounded), or `None` when `token_count == 0`.
    pub compression_ratio: Option<f64>,
}

/// Summary compression-quality diagnostics for one session (`lcm_doctor`'s `summary_quality`) —
/// the port of `_summary_quality_stats` (`LCM:tools.py:1449`).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SummaryQuality {
    /// Total summary nodes for the session.
    pub total_nodes: i64,
    /// The session the stats cover.
    pub session_id: String,
    /// Sum of summarized-span tokens.
    pub total_source_tokens: i64,
    /// Sum of summary tokens.
    pub total_summary_tokens: i64,
    /// `total_source_tokens / total_summary_tokens` (rounded), or `0.0` when no summaries.
    pub overall_compression_ratio: f64,
    /// The extreme-ratio threshold (a fixed `400`).
    pub extreme_ratio_threshold: i64,
    /// Nodes whose source/summary ratio is `>= 400` (degraded fallback summaries).
    pub extreme_ratio_nodes: i64,
    /// Nodes summarizing a very large span into a tiny summary (`source >= 100000 AND token < 500`).
    pub tiny_large_source_nodes: i64,
    /// Up to five worst-offending nodes (inspect via `lcm_expand`).
    pub worst_nodes: Vec<WorstNode>,
}

/// Source-attribution bucket counts (`lcm_doctor`'s `source_lineage_hygiene`) — the port of
/// `get_source_stats` (`LCM:store.py:586`).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SourceStats {
    /// Total messages in scope.
    pub messages_total: i64,
    /// Messages with a concrete (non-unknown, non-blank) source.
    pub attributed_messages: i64,
    /// Messages normalized to the explicit `"unknown"` bucket.
    pub normalized_unknown_messages: i64,
    /// Legacy rows with a NULL/blank source (pre-normalization).
    pub legacy_blank_source_messages: i64,
    /// `normalized_unknown_messages + legacy_blank_source_messages`.
    pub effective_unknown_messages: i64,
}

/// Lifecycle/session fragmentation diagnostics (`lcm_doctor`'s `lifecycle_fragmentation`) — the
/// in-database portion of `get_fragmentation_stats` (`LCM:lifecycle_state.py:337`). The external host
/// `state_db` comparison is intentionally omitted (the daemon has no separate host sessions DB).
/// Read-only: reports mismatches without inferring corruption or rewriting any state.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct LifecycleFragmentation {
    /// Always `true` — this diagnostic never mutates.
    pub read_only: bool,
    /// Rows in `lcm_lifecycle_state`.
    pub lifecycle_rows: i64,
    /// Total `messages` rows.
    pub messages_total: i64,
    /// Total `summary_nodes` rows.
    pub summary_nodes_total: i64,
    /// Distinct sessions seen in `messages`.
    pub distinct_message_sessions: i64,
    /// Distinct sessions seen in `summary_nodes`.
    pub distinct_node_sessions: i64,
    /// Distinct sessions seen in messages or nodes (the union).
    pub distinct_lcm_any_sessions: i64,
    /// Distinct `current_session_id` values in lifecycle state.
    pub lifecycle_current_sessions: i64,
    /// Distinct `last_finalized_session_id` values in lifecycle state.
    pub lifecycle_last_finalized_sessions: i64,
    /// Lifecycle `current` sessions absent from `messages`.
    pub lifecycle_current_missing_in_messages: i64,
    /// Lifecycle `current` sessions absent from `summary_nodes`.
    pub lifecycle_current_missing_in_nodes: i64,
    /// Lifecycle `current` sessions absent from messages and nodes.
    pub lifecycle_current_missing_in_lcm_any: i64,
    /// Lifecycle `last_finalized` sessions absent from `messages`.
    pub lifecycle_last_finalized_missing_in_messages: i64,
    /// Lifecycle `last_finalized` sessions absent from `summary_nodes`.
    pub lifecycle_last_finalized_missing_in_nodes: i64,
    /// Lifecycle `last_finalized` sessions absent from messages and nodes.
    pub lifecycle_last_finalized_missing_in_lcm_any: i64,
    /// Message sessions with no lifecycle `current` reference.
    pub message_sessions_without_lifecycle_current: i64,
    /// Message sessions with no lifecycle reference at all (current or finalized).
    pub message_sessions_without_lifecycle_reference: i64,
    /// Node sessions with no lifecycle reference at all (current or finalized).
    pub node_sessions_without_lifecycle_reference: i64,
}

impl LifecycleFragmentation {
    /// Whether these diagnostics should be treated as warning evidence — the port of
    /// `has_lifecycle_fragmentation` (`LCM:diagnostics.py:33`), minus the omitted `state_db` keys.
    pub fn is_fragmented(&self) -> bool {
        self.lifecycle_current_missing_in_lcm_any > 0
            || self.lifecycle_last_finalized_missing_in_lcm_any > 0
            || (self.lifecycle_rows > 0
                && (self.message_sessions_without_lifecycle_reference > 0
                    || self.node_sessions_without_lifecycle_reference > 0))
    }
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

    /// Delete the volatile, uncompacted tail (`store_id > after_store_id`) for a session — used by
    /// the rehydration reconcile to rebuild the live tail from the replayed conversation without
    /// duplicating rows. Messages at or below the frontier are immutable (referenced by D0 nodes).
    pub fn delete_messages_after(&self, session_id: &str, after_store_id: i64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1 AND store_id > ?2",
            params![session_id, after_store_id],
        )?;
        Ok(())
    }

    /// Rewrite a single message's `content` in place, preserving its `store_id` (the FTS update
    /// trigger keeps the shadow in sync). The §9.1 transcript-GC path — the one path that mutates a
    /// `messages` row after insert.
    pub fn update_message_content(&self, store_id: i64, content: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "UPDATE messages SET content = ?2 WHERE store_id = ?1",
            params![store_id, content],
        )?;
        Ok(())
    }

    /// Transcript-GC candidates (§9.1): summarized rows (`store_id <= max_store_id`) that still carry
    /// an *un-GC'd* externalized-payload placeholder inline, for `gc_externalized_tool_result`.
    pub fn messages_to_gc(&self, session_id: &str, max_store_id: i64) -> Result<Vec<MessageRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages \
             WHERE session_id = ?1 AND store_id <= ?2 \
             AND content LIKE '%Externalized %' AND content NOT LIKE '%GC''d externalized%' \
             ORDER BY store_id ASC",
        )?;
        let rows = stmt
            .query_map(params![session_id, max_store_id], map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
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

    /// FTS5 transcript search returning candidate rows + `rank` (lower = better), ordered by rank.
    /// `match_query` must already be an FTS5-safe expression (see `search::sanitize_fts5_query`).
    /// Filtering/final ordering/snippeting is the caller's job (`search.rs`).
    pub fn search_messages_fts(
        &self,
        match_query: &str,
        filter: &MessageFilter<'_>,
        limit: i64,
    ) -> Result<Vec<MessageHit>> {
        let mut sql = String::from(
            "SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id, \
             m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, f.rank \
             FROM messages_fts f JOIN messages m ON m.store_id = f.rowid \
             WHERE f.content MATCH ?",
        );
        let mut args: Vec<Value> = vec![Value::Text(match_query.to_string())];
        push_message_filters(&mut sql, &mut args, filter);
        sql.push_str(" ORDER BY f.rank LIMIT ?");
        args.push(Value::Integer(limit));

        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), |r| {
                Ok(MessageHit {
                    row: map_message(r)?,
                    rank: r.get::<_, f64>(10)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// LIKE-fallback transcript search (§11.2) for queries FTS5 can't safely handle (CJK/emoji/risky
    /// ASCII). Returns rows matching ANY term (`content LIKE '%term%' ESCAPE '\'`); the caller scores.
    pub fn search_messages_like(
        &self,
        like_terms: &[String],
        filter: &MessageFilter<'_>,
        limit: i64,
    ) -> Result<Vec<MessageRow>> {
        if like_terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = String::from(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages m WHERE (",
        );
        let mut args: Vec<Value> = Vec::new();
        for (i, term) in like_terms.iter().enumerate() {
            if i > 0 {
                sql.push_str(" OR ");
            }
            sql.push_str("content LIKE ? ESCAPE '\\'");
            args.push(Value::Text(format!("%{}%", like_escape(term))));
        }
        sql.push(')');
        push_message_filters(&mut sql, &mut args, filter);
        sql.push_str(" ORDER BY timestamp DESC LIMIT ?");
        args.push(Value::Integer(limit));

        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// FTS5 search over the summary DAG, returning nodes + `rank` (lower = better), ordered by rank.
    pub fn search_nodes_fts(
        &self,
        match_query: &str,
        session_id: &str,
        limit: i64,
    ) -> Result<Vec<NodeHit>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT n.node_id, n.session_id, n.depth, n.summary, n.token_count, \
             n.source_token_count, n.source_ids, n.source_type, n.created_at, n.earliest_at, \
             n.latest_at, n.expand_hint, f.rank \
             FROM nodes_fts f JOIN summary_nodes n ON n.node_id = f.rowid \
             WHERE f.summary MATCH ?1 AND n.session_id = ?2 ORDER BY f.rank LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![match_query, session_id, limit], |r| {
                Ok(NodeHit {
                    node: map_node(r)?,
                    rank: r.get::<_, f64>(12)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// A page of a session's transcript (oldest first), for `lcm_load_session` / `lcm_expand`.
    pub fn messages_page(
        &self,
        session_id: &str,
        offset: i64,
        limit: i64,
    ) -> Result<Vec<MessageRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages \
             WHERE session_id = ?1 ORDER BY store_id ASC LIMIT ?3 OFFSET ?2",
        )?;
        let rows = stmt
            .query_map(params![session_id, offset, limit], map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// A `store_id`-cursored page of a session's transcript (oldest first, `store_id > after`), for
    /// `lcm_load_session`. Pass `after = None` (or `0`) for the first page; fetch `limit + 1` to
    /// detect `has_more` (§10.2).
    pub fn load_session_page(
        &self,
        session_id: &str,
        after: Option<i64>,
        limit: i64,
    ) -> Result<Vec<MessageRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages \
             WHERE session_id = ?1 AND store_id > ?2 ORDER BY store_id ASC LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![session_id, after.unwrap_or(0), limit], map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetch a set of messages by `store_id` (any session), returned in ascending `store_id` order —
    /// the lossless recovery path for a D0 node's `source_ids` (`lcm_expand`).
    pub fn get_messages(&self, store_ids: &[i64]) -> Result<Vec<MessageRow>> {
        if store_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = String::from(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages WHERE store_id IN (",
        );
        let mut args: Vec<Value> = Vec::with_capacity(store_ids.len());
        for (i, id) in store_ids.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('?');
            args.push(Value::Integer(*id));
        }
        sql.push_str(") ORDER BY store_id ASC");
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), map_message)?
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

    /// Bank-wide row counts for `lcm_status`/`lcm_doctor` (the base table + its FTS shadow, so an
    /// out-of-sync FTS index is detectable).
    pub fn table_counts(&self) -> Result<StoreCounts> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let one = |sql: &str| conn.query_row(sql, [], |r| r.get::<_, i64>(0));
        Ok(StoreCounts {
            messages: one("SELECT COUNT(*) FROM messages")?,
            messages_fts: one("SELECT COUNT(*) FROM messages_fts")?,
            nodes: one("SELECT COUNT(*) FROM summary_nodes")?,
            nodes_fts: one("SELECT COUNT(*) FROM nodes_fts")?,
        })
    }

    /// SQLite `PRAGMA integrity_check` first row (`"ok"` when healthy) — `lcm_doctor`.
    pub fn integrity_check(&self) -> Result<String> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let s = conn.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?;
        Ok(s)
    }

    /// Count summary nodes whose `source_type='nodes'` reference a missing child node (orphans) —
    /// `lcm_doctor`'s `orphaned_dag_nodes` check.
    pub fn orphaned_node_count(&self) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(
            "SELECT COUNT(*) FROM summary_nodes p, json_each(p.source_ids) j \
             WHERE p.source_type = 'nodes' \
             AND NOT EXISTS (SELECT 1 FROM summary_nodes c WHERE c.node_id = CAST(j.value AS INTEGER))",
            [],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// The presence of the schema's core objects — `lcm_doctor`'s `schema_core_tables` check (§10.6).
    pub fn schema_health(&self) -> Result<SchemaHealth> {
        const CORE: [&str; 7] = [
            "messages",
            "summary_nodes",
            "lcm_lifecycle_state",
            "metadata",
            "lcm_migration_state",
            "messages_fts",
            "nodes_fts",
        ];
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut present = Vec::new();
        let mut missing = Vec::new();
        for name in CORE {
            let exists: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name = ?1)",
                [name],
                |r| r.get::<_, i64>(0).map(|n| n != 0),
            )?;
            if exists {
                present.push(name.to_string());
            } else {
                missing.push(name.to_string());
            }
        }
        let schema_version = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = 'schema_version'",
                [],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .and_then(|s| s.parse::<i64>().ok());
        Ok(SchemaHealth {
            present,
            missing,
            schema_version,
        })
    }

    /// SQLite storage posture (journal mode, `quick_check`, backing path, size) — `lcm_doctor`'s
    /// `sqlite_storage` check (§10.6).
    pub fn storage_posture(&self) -> Result<StoragePosture> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let journal_mode = conn
            .query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))
            .unwrap_or_default();
        let quick_check = conn
            .query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))
            .unwrap_or_else(|_| "unknown".to_string());
        let page_count: i64 = conn
            .query_row("PRAGMA page_count", [], |r| r.get(0))
            .unwrap_or(0);
        let page_size: i64 = conn
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .unwrap_or(0);
        // `pragma_database_list` exposes the main db's backing file (empty for an in-memory bank),
        // so the check reports the path without the `Store` having to carry it.
        let database_path = conn
            .query_row(
                "SELECT file FROM pragma_database_list WHERE name = 'main'",
                [],
                |r| r.get::<_, String>(0),
            )
            .unwrap_or_default();
        let in_memory = database_path.is_empty();
        Ok(StoragePosture {
            database_path,
            in_memory,
            journal_mode,
            quick_check,
            database_size_bytes: page_count * page_size,
        })
    }

    /// Summary compression-quality diagnostics for one session — `lcm_doctor`'s `summary_quality`
    /// check (§10.6), the port of `_summary_quality_stats` (`LCM:tools.py:1449`).
    pub fn summary_quality_stats(&self, session_id: &str) -> Result<SummaryQuality> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let (
            total_nodes,
            total_source_tokens,
            total_summary_tokens,
            tiny_large_source_nodes,
            extreme_ratio_nodes,
        ) = conn.query_row(
            "SELECT COUNT(*), \
             COALESCE(SUM(source_token_count), 0), \
             COALESCE(SUM(token_count), 0), \
             COALESCE(SUM(CASE WHEN source_token_count >= 100000 AND token_count < 500 \
                          THEN 1 ELSE 0 END), 0), \
             COALESCE(SUM(CASE WHEN token_count > 0 \
                          AND CAST(source_token_count AS REAL) / token_count >= 400 \
                          THEN 1 ELSE 0 END), 0) \
             FROM summary_nodes WHERE session_id = ?1",
            [session_id],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            },
        )?;
        let overall_compression_ratio = if total_summary_tokens > 0 {
            round1(total_source_tokens as f64 / total_summary_tokens as f64)
        } else {
            0.0
        };
        let mut stmt = conn.prepare(
            "SELECT node_id, session_id, depth, token_count, source_token_count \
             FROM summary_nodes \
             WHERE session_id = ?1 AND source_token_count > 0 \
             ORDER BY \
                 CASE WHEN token_count <= 0 THEN 1 ELSE 0 END DESC, \
                 CASE WHEN token_count > 0 \
                      THEN CAST(source_token_count AS REAL) / token_count \
                      ELSE source_token_count END DESC \
             LIMIT 5",
        )?;
        let worst_nodes = stmt
            .query_map([session_id], |r| {
                let token_count: i64 = r.get(3)?;
                let source_token_count: i64 = r.get(4)?;
                let compression_ratio = if token_count > 0 {
                    Some(round1(source_token_count as f64 / token_count as f64))
                } else {
                    None
                };
                Ok(WorstNode {
                    node_id: r.get(0)?,
                    session_id: r.get(1)?,
                    depth: r.get(2)?,
                    source_token_count,
                    token_count,
                    compression_ratio,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(SummaryQuality {
            total_nodes,
            session_id: session_id.to_string(),
            total_source_tokens,
            total_summary_tokens,
            overall_compression_ratio,
            extreme_ratio_threshold: 400,
            extreme_ratio_nodes,
            tiny_large_source_nodes,
            worst_nodes,
        })
    }

    /// Source-attribution bucket counts — `lcm_doctor`'s `source_lineage_hygiene` check (§10.6), the
    /// port of `get_source_stats` (`LCM:store.py:586`). `session_id = None` is bank-wide (the doctor's
    /// call); `Some` scopes to one session.
    pub fn source_stats(&self, session_id: Option<&str>) -> Result<SourceStats> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let where_clause = if session_id.is_some() {
            "WHERE session_id = ?1"
        } else {
            ""
        };
        let sql = format!(
            "SELECT COUNT(*), \
             COALESCE(SUM(CASE WHEN source = 'unknown' THEN 1 ELSE 0 END), 0), \
             COALESCE(SUM(CASE WHEN source IS NULL OR TRIM(source) = '' THEN 1 ELSE 0 END), 0), \
             COALESCE(SUM(CASE WHEN source IS NOT NULL AND TRIM(source) != '' \
                          AND source != 'unknown' THEN 1 ELSE 0 END), 0) \
             FROM messages {where_clause}"
        );
        let map = |r: &Row<'_>| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
            ))
        };
        let (messages_total, normalized_unknown_messages, legacy_blank_source_messages, attributed_messages) =
            match session_id {
                Some(sid) => conn.query_row(&sql, [sid], map)?,
                None => conn.query_row(&sql, [], map)?,
            };
        Ok(SourceStats {
            messages_total,
            attributed_messages,
            normalized_unknown_messages,
            legacy_blank_source_messages,
            effective_unknown_messages: normalized_unknown_messages + legacy_blank_source_messages,
        })
    }

    /// Lifecycle/session fragmentation diagnostics — `lcm_doctor`'s `lifecycle_fragmentation` check
    /// (§10.6), the in-database portion of `get_fragmentation_stats` (`LCM:lifecycle_state.py:337`).
    pub fn lifecycle_fragmentation_stats(&self) -> Result<LifecycleFragmentation> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let session_set = |sql: &str| -> rusqlite::Result<HashSet<String>> {
            let mut stmt = conn.prepare(sql)?;
            let rows = stmt.query_map([], |r| r.get::<_, Option<String>>(0))?;
            let mut set = HashSet::new();
            for row in rows {
                if let Some(v) = row? {
                    if !v.is_empty() {
                        set.insert(v);
                    }
                }
            }
            Ok(set)
        };
        let message_sessions = session_set(
            "SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL",
        )?;
        let node_sessions = session_set(
            "SELECT DISTINCT session_id FROM summary_nodes WHERE session_id IS NOT NULL",
        )?;
        let lifecycle_current = session_set(
            "SELECT DISTINCT current_session_id FROM lcm_lifecycle_state \
             WHERE current_session_id IS NOT NULL",
        )?;
        let lifecycle_finalized = session_set(
            "SELECT DISTINCT last_finalized_session_id FROM lcm_lifecycle_state \
             WHERE last_finalized_session_id IS NOT NULL",
        )?;
        let lcm_any: HashSet<String> = message_sessions.union(&node_sessions).cloned().collect();
        let referenced: HashSet<String> =
            lifecycle_current.union(&lifecycle_finalized).cloned().collect();
        let count = |sql: &str| conn.query_row(sql, [], |r| r.get::<_, i64>(0));
        let diff = |a: &HashSet<String>, b: &HashSet<String>| a.difference(b).count() as i64;
        Ok(LifecycleFragmentation {
            read_only: true,
            lifecycle_rows: count("SELECT COUNT(*) FROM lcm_lifecycle_state")?,
            messages_total: count("SELECT COUNT(*) FROM messages")?,
            summary_nodes_total: count("SELECT COUNT(*) FROM summary_nodes")?,
            distinct_message_sessions: message_sessions.len() as i64,
            distinct_node_sessions: node_sessions.len() as i64,
            distinct_lcm_any_sessions: lcm_any.len() as i64,
            lifecycle_current_sessions: lifecycle_current.len() as i64,
            lifecycle_last_finalized_sessions: lifecycle_finalized.len() as i64,
            lifecycle_current_missing_in_messages: diff(&lifecycle_current, &message_sessions),
            lifecycle_current_missing_in_nodes: diff(&lifecycle_current, &node_sessions),
            lifecycle_current_missing_in_lcm_any: diff(&lifecycle_current, &lcm_any),
            lifecycle_last_finalized_missing_in_messages: diff(&lifecycle_finalized, &message_sessions),
            lifecycle_last_finalized_missing_in_nodes: diff(&lifecycle_finalized, &node_sessions),
            lifecycle_last_finalized_missing_in_lcm_any: diff(&lifecycle_finalized, &lcm_any),
            message_sessions_without_lifecycle_current: diff(&message_sessions, &lifecycle_current),
            message_sessions_without_lifecycle_reference: diff(&message_sessions, &referenced),
            node_sessions_without_lifecycle_reference: diff(&node_sessions, &referenced),
        })
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

/// Round to one decimal place (matching the Python diagnostics' `round(x, 1)`).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
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

/// Append the optional `MessageFilter` clauses (and their bind values) to a search query whose
/// table alias is `m`. Filters use anonymous `?` placeholders bound in append order.
fn push_message_filters(sql: &mut String, args: &mut Vec<Value>, filter: &MessageFilter<'_>) {
    if let Some(session) = filter.session {
        sql.push_str(" AND m.session_id = ?");
        args.push(Value::Text(session.to_string()));
    }
    if let Some(role) = filter.role {
        sql.push_str(" AND m.role = ?");
        args.push(Value::Text(role.to_string()));
    }
    if let Some(source) = filter.source {
        sql.push_str(" AND m.source = ?");
        args.push(Value::Text(source.to_string()));
    }
    if let Some(from) = filter.time_from {
        sql.push_str(" AND m.timestamp >= ?");
        args.push(Value::Real(from));
    }
    if let Some(to) = filter.time_to {
        sql.push_str(" AND m.timestamp <= ?");
        args.push(Value::Real(to));
    }
}

/// Escape LIKE wildcards in a literal term (`\` is the ESCAPE char in the LIKE queries above).
fn like_escape(term: &str) -> String {
    term.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
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
