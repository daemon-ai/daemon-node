// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The LCM store (`daemon-context-lcm-port-spec.md` §4): one SQLite file per bank holding the
//! lossless `messages` transcript, the summary DAG, and the lifecycle frontier.
//!
//! Concurrency follows `daemon-mnemosyne`: a serialized [`Mutex<Connection>`] in WAL mode with
//! `synchronous=FULL` (LCM's lossless contract wants per-commit durability — §4.1). The spec's
//! dedicated store-actor (§4.7) is a later refinement; the seam above it does not change.

pub mod schema;

use crate::error::Result;
use crate::search::{SortMode, AGE_DECAY_RATE};
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection, Row};
use rusqlite_migration::{Migrations, M};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

/// The schema-migration ladder, gated by `PRAGMA user_version` (rusqlite_migration). `M1` is the
/// full §4 schema; `M2` adds the lifecycle maintenance/rollover/reset timestamps + the
/// finalized-session index; future schema changes append an `M::up("…")`. Pragmas (WAL +
/// `synchronous=FULL`) are applied in [`Store::init`] *before* `to_latest`, since `journal_mode`
/// cannot change inside the migration transaction.
static MIGRATIONS: LazyLock<Migrations<'static>> =
    LazyLock::new(|| Migrations::new(vec![M::up(schema::SCHEMA), M::up(schema::MIGRATION_V2)]));

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
    /// The SQL-side `snippet(messages_fts, …, '>>>', '<<<', '...', 40)` excerpt (`LCM:store.py:739`).
    pub snippet: String,
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

/// The outcome of one external-content FTS repair pass (`repair_external_content_fts`,
/// `LCM:db_bootstrap.py:524-569`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct FtsRepair {
    /// The FTS table repaired.
    pub table: String,
    /// Whether the index was dropped and rebuilt from its content table.
    pub rebuilt: bool,
    /// Whether any sync trigger was missing and recreated.
    pub triggers_recreated: bool,
    /// Whether the rebuild hit a disk-space-class write error and the index was dropped instead
    /// (search degrades to LIKE-only until a later repair succeeds — `LCM:db_bootstrap.py:533-545`).
    pub degraded: bool,
}

/// One table's verdict from the read-only `/lcm doctor repair` scan (`_scan_fts_repair`,
/// `LCM:command.py:665-701`): structural + unthrottled deep check plus the row counts an operator
/// compares by eye.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FtsRepairScan {
    /// The FTS table scanned.
    pub table: String,
    /// Whether a repair pass would rebuild this index.
    pub needs_repair: bool,
    /// `COUNT(*)` of the content table (`None` when the query itself failed).
    pub content_rows: Option<i64>,
    /// `COUNT(*)` of the FTS table (`None` when unreadable — itself repair evidence).
    pub fts_rows: Option<i64>,
}

/// One FTS5 deep integrity-check verdict (`check_external_content_fts_integrity`,
/// `LCM:db_bootstrap.py:424-463`): `pass`, `fail`, or `unchecked` (read-only database).
#[derive(Clone, Debug, serde::Serialize)]
pub struct FtsIntegrity {
    /// The FTS table checked.
    pub table: String,
    /// `pass` | `fail` | `unchecked`.
    pub status: String,
    /// Human-readable evidence (`ok`, or the SQLite error).
    pub detail: String,
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

/// One `lcm_lifecycle_state` row (`LifecycleState`, `LCM:lifecycle_state.py:22`): which session a
/// logical conversation is bound to, what was last finalized, the compaction frontier markers, any
/// deferred-compaction debt, and the lifecycle timestamps.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct LifecycleRow {
    /// The logical conversation this row keys (defaults to the first bound session id).
    pub conversation_id: String,
    /// The currently bound session, if any.
    pub current_session_id: Option<String>,
    /// The most recently finalized session, if any.
    pub last_finalized_session_id: Option<String>,
    /// The active compaction frontier (highest `store_id` covered by a D0 node).
    pub current_frontier_store_id: i64,
    /// The frontier as of the last finalize (monotonic high-water mark).
    pub last_finalized_frontier_store_id: i64,
    /// Deferred-compaction debt kind, if any.
    pub debt_kind: Option<String>,
    /// Deferred-compaction debt size estimate (tokens).
    pub debt_size_estimate: i64,
    /// When the current session was bound.
    pub current_bound_at: Option<f64>,
    /// When the last finalize ran.
    pub last_finalized_at: Option<f64>,
    /// When the debt fields last changed.
    pub debt_updated_at: Option<f64>,
    /// When maintenance last ran (attempted) for this conversation.
    pub last_maintenance_attempt_at: Option<f64>,
    /// When the conversation last rolled over to a new session.
    pub last_rollover_at: Option<f64>,
    /// When retained state was last reset (`/new`).
    pub last_reset_at: Option<f64>,
    /// Last write to this row.
    pub updated_at: f64,
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

/// The default deep FTS integrity-check throttle in hours (the
/// `fts_integrity_check_interval_hours` config default — `LCM:db_bootstrap.py:331-348`).
const DEFAULT_FTS_CHECK_INTERVAL_HOURS: f64 = 24.0;

/// The serialized LCM store.
pub struct Store {
    conn: Mutex<Connection>,
    /// Hours between deep FTS integrity-checks on the throttled repair path (`0` checks every
    /// time, negative never deep-checks) — injected from
    /// [`LcmConfig::fts_integrity_check_interval_hours`](crate::config::LcmConfig).
    fts_check_interval_hours: f64,
    /// Whether an FTS rebuild hit a disk-space-class write error and dropped the index instead
    /// (low-disk degradation, `LCM:db_bootstrap.py:533-545`): search routes to LIKE-only while
    /// set. Cleared when a later repair pass (next open, or a forced `/lcm doctor repair apply`)
    /// rebuilds cleanly.
    degraded: AtomicBool,
}

impl Store {
    /// Open (or create) the store at `path`, applying the v4 schema (the default FTS throttle).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_at(Some(path.as_ref()), DEFAULT_FTS_CHECK_INTERVAL_HOURS)
    }

    /// Open an in-memory store (tests / ephemeral nodes) with the default FTS throttle.
    pub fn open_in_memory() -> Result<Self> {
        Self::open_at(None, DEFAULT_FTS_CHECK_INTERVAL_HOURS)
    }

    /// Open the store at `path` (in-memory when `None`) with a configured deep FTS
    /// integrity-check interval — the engine's construction seam (the crate reads no environment;
    /// the host injects the interval through the config).
    pub fn open_at(path: Option<&Path>, fts_check_interval_hours: f64) -> Result<Self> {
        let conn = match path {
            Some(path) => {
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                Connection::open(path)?
            }
            None => Connection::open_in_memory()?,
        };
        Self::init(conn, fts_check_interval_hours)
    }

    fn init(mut conn: Connection, fts_check_interval_hours: f64) -> Result<Self> {
        // §4.1 pragmas — WAL + FULL durability (LCM is lossless), generous lock wait, bounded WAL.
        // Applied OUTSIDE the migration ladder: `to_latest` runs in a transaction and `journal_mode`
        // cannot change inside one.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA busy_timeout=30000;
             PRAGMA wal_autocheckpoint=500;
             PRAGMA journal_size_limit=67108864;
             PRAGMA mmap_size=268435456;",
        )?;
        // `PRAGMA user_version` is the schema authority. Mirror it into `metadata.schema_version` so
        // the value stays readable from SQL (e.g. ad-hoc inspection); `lcm_migration_state` is no
        // longer hand-stamped now that the ladder owns versioning.
        MIGRATIONS.to_latest(&mut conn)?;
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?1)",
            params![schema::SCHEMA_VERSION.to_string()],
        )?;
        // Startup FTS hygiene (`ensure_external_content_fts`, `LCM:db_bootstrap.py:572-577`):
        // structurally verify + (throttled) deep-check each external-content index, rebuilding a
        // broken one from its content table so a corrupted index heals on open. A rebuild that
        // hits a full disk degrades that index to LIKE-only search instead of failing the open.
        let now = unix_now();
        let mut degraded = false;
        for spec in [&schema::MESSAGES_FTS, &schema::NODES_FTS] {
            degraded |=
                repair_fts_locked(&conn, spec, now, Some(fts_check_interval_hours))?.degraded;
        }
        Ok(Self {
            conn: Mutex::new(conn),
            fts_check_interval_hours,
            degraded: AtomicBool::new(degraded),
        })
    }

    /// Whether FTS is currently degraded to LIKE-only search (a rebuild hit a disk-space-class
    /// write error and the index was dropped). Recovers when a later repair pass — the next
    /// [`Store::open`], or a forced [`Store::repair_fts`] — rebuilds cleanly.
    pub fn is_degraded(&self) -> bool {
        self.degraded.load(Ordering::Relaxed)
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
                // `_normalize_source_value` (`LCM:store.py:69-71`): strip, blank -> `unknown`.
                let source = match m.source.trim() {
                    "" => "unknown",
                    s => s,
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

    /// Transcript-GC candidates (§9.1): summarized rows (`store_id <= max_store_id`) that still carry
    /// an *un-GC'd* externalized-payload placeholder inline, for `gc_externalized_tool_result`.
    /// Only unpinned `tool` rows qualify (`_maybe_gc_compacted_tool_results`,
    /// `LCM:engine.py:3448-3453`) — user/assistant rows are never GC'd.
    pub fn messages_to_gc(&self, session_id: &str, max_store_id: i64) -> Result<Vec<MessageRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages \
             WHERE session_id = ?1 AND store_id <= ?2 \
             AND role = 'tool' AND pinned = 0 \
             AND content LIKE '%Externalized %' AND content NOT LIKE '%GC''d externalized%' \
             ORDER BY store_id ASC",
        )?;
        let rows = stmt
            .query_map(params![session_id, max_store_id], map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Rewrite one unpinned tool-result row to a compact GC placeholder, updating its cached token
    /// estimate in the same write (`gc_externalized_tool_result`, `LCM:store.py:381-405`). The
    /// role/pinned/idempotence guards re-run under the connection lock, so a row pinned (or already
    /// rewritten) after candidate selection is left untouched. The placeholder's token estimate is
    /// the caller's job (the store holds no tokenizer). Returns whether the row was rewritten.
    pub fn gc_externalized_tool_result(
        &self,
        store_id: i64,
        placeholder: &str,
        placeholder_tokens: i64,
    ) -> Result<bool> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let row: Option<(String, i64, Option<String>)> = conn
            .query_row(
                "SELECT role, pinned, content FROM messages WHERE store_id = ?1",
                [store_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        let Some((role, pinned, content)) = row else {
            return Ok(false);
        };
        if role != "tool" || pinned != 0 || content.as_deref() == Some(placeholder) {
            return Ok(false);
        }
        conn.execute(
            "UPDATE messages SET content = ?2, token_estimate = ?3 WHERE store_id = ?1",
            params![store_id, placeholder, placeholder_tokens],
        )?;
        Ok(true)
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
            .query_map(params![sanitized, session_id, limit], |r| {
                r.get::<_, i64>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// One page of FTS5 transcript candidates + `rank` + SQL snippet, ordered by the sort mode's
    /// SQL `ORDER BY` (`Store.search`, `LCM:store.py:702-745`). `match_query` must already be an
    /// FTS5-safe expression (see `search::sanitize_fts5_query`); the widening ladder, directness
    /// re-ranking, and final truncation live in `search.rs`.
    pub fn search_messages_fts(
        &self,
        match_query: &str,
        filter: &MessageFilter<'_>,
        sort: SortMode,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MessageHit>> {
        let mut sql = String::from(
            "SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id, \
             m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, f.rank, \
             snippet(messages_fts, 0, '>>>', '<<<', '...', 40) \
             FROM messages_fts f JOIN messages m ON m.store_id = f.rowid \
             WHERE f.content MATCH ?",
        );
        let mut args: Vec<Value> = vec![Value::Text(match_query.to_string())];
        push_message_filters(&mut sql, &mut args, filter);
        sql.push_str(&format!(
            " ORDER BY {} LIMIT ? OFFSET ?",
            message_search_order_by(sort)
        ));
        args.push(Value::Integer(limit));
        args.push(Value::Integer(offset));

        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), |r| {
                Ok(MessageHit {
                    row: map_message(r)?,
                    rank: r.get::<_, f64>(10)?,
                    snippet: r.get::<_, Option<String>>(11)?.unwrap_or_default(),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// LIKE-fallback transcript candidates with **no** `ORDER BY` (`LCM:store.py:968-976`): the
    /// non-recency sorts fetch one unordered batch and rank purely in the caller.
    pub fn search_messages_like_unordered(
        &self,
        like_terms: &[String],
        filter: &MessageFilter<'_>,
        limit: i64,
    ) -> Result<Vec<MessageRow>> {
        if like_terms.is_empty() {
            return Ok(Vec::new());
        }
        let (sql, mut args) = like_message_where(like_terms, filter);
        let sql = format!(
            "SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id, \
             m.tool_calls, m.tool_name, m.timestamp, m.token_estimate FROM messages m \
             WHERE {sql} LIMIT ?"
        );
        args.push(Value::Integer(limit));
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// One page of LIKE-fallback candidates under the recency-sort SQL `ORDER BY` — newest first,
    /// then role bias, SQL-side term-hit score, SQL-side directness, `store_id`
    /// (`LCM:store.py:835-902`). `collapse_risky_repeats` counts each term at most once (risky-ASCII
    /// queries); the caller drives the paging/tie-continuation ladder.
    pub fn search_messages_like_recency(
        &self,
        like_terms: &[String],
        phrases: &[String],
        collapse_risky_repeats: bool,
        filter: &MessageFilter<'_>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MessageRow>> {
        if like_terms.is_empty() {
            return Ok(Vec::new());
        }
        let (where_sql, mut args) = like_message_where(like_terms, filter);
        let (order_sql, order_args) =
            like_recency_order_by(like_terms, phrases, collapse_risky_repeats);
        let sql = format!(
            "SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id, \
             m.tool_calls, m.tool_name, m.timestamp, m.token_estimate FROM messages m \
             WHERE {where_sql} {order_sql} LIMIT ? OFFSET ?"
        );
        args.extend(order_args);
        args.push(Value::Integer(limit));
        args.push(Value::Integer(offset));
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// One page of FTS5 summary-DAG candidates + `rank`, ordered by the node sort mode's SQL
    /// `ORDER BY` (`SummaryDAG.search`, `LCM:dag.py:350-376`). `session = None` searches every
    /// session in the bank.
    pub fn search_nodes_fts(
        &self,
        match_query: &str,
        session: Option<&str>,
        sort: SortMode,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<NodeHit>> {
        let mut sql = String::from(
            "SELECT n.node_id, n.session_id, n.depth, n.summary, n.token_count, \
             n.source_token_count, n.source_ids, n.source_type, n.created_at, n.earliest_at, \
             n.latest_at, n.expand_hint, f.rank \
             FROM nodes_fts f JOIN summary_nodes n ON n.node_id = f.rowid \
             WHERE f.summary MATCH ?",
        );
        let mut args: Vec<Value> = vec![Value::Text(match_query.to_string())];
        if let Some(session) = session {
            sql.push_str(" AND n.session_id = ?");
            args.push(Value::Text(session.to_string()));
        }
        sql.push_str(&format!(
            " ORDER BY {} LIMIT ? OFFSET ?",
            node_search_order_by(sort)
        ));
        args.push(Value::Integer(limit));
        args.push(Value::Integer(offset));
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), |r| {
                Ok(NodeHit {
                    node: map_node(r)?,
                    rank: r.get::<_, f64>(12)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// One unordered page of LIKE-fallback summary-node candidates
    /// (`SummaryDAG._search_like`, `LCM:dag.py:427-451`); the caller scores and sorts.
    pub fn search_nodes_like(
        &self,
        like_terms: &[String],
        session: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SummaryNode>> {
        if like_terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = format!("{NODE_SELECT} WHERE summary IS NOT NULL");
        let mut args: Vec<Value> = Vec::new();
        if let Some(session) = session {
            sql.push_str(" AND session_id = ?");
            args.push(Value::Text(session.to_string()));
        }
        sql.push_str(" AND (");
        for (i, term) in like_terms.iter().enumerate() {
            if i > 0 {
                sql.push_str(" OR ");
            }
            sql.push_str("summary LIKE ? ESCAPE '\\'");
            args.push(Value::Text(format!("%{}%", like_escape(term))));
        }
        sql.push_str(") LIMIT ? OFFSET ?");
        args.push(Value::Integer(limit));
        args.push(Value::Integer(offset));
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), map_node)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Whether any raw message in `node_id`'s descendant lineage matches `source`
    /// (`SummaryDAG._node_matches_source`, `LCM:dag.py:492-537`): a recursive CTE walks
    /// `source_ids` down through child nodes to raw `messages` rows, with the `unknown` bucket
    /// also matching legacy NULL/blank sources. Results are memoized in `cache` per search call.
    pub fn node_matches_source(
        &self,
        node_id: i64,
        source: &str,
        cache: &mut std::collections::HashMap<i64, bool>,
    ) -> Result<bool> {
        if source.is_empty() {
            return Ok(true);
        }
        let normalized = match source.trim() {
            "" => "unknown",
            s => s,
        };
        if let Some(&matched) = cache.get(&node_id) {
            return Ok(matched);
        }
        let blank = legacy_blank_source_clause("m.source");
        let sql = format!(
            "WITH RECURSIVE source_walk(source_type, source_id) AS (\n\
                 SELECT n.source_type, CAST(j.value AS INTEGER)\n\
                 FROM summary_nodes n, json_each(n.source_ids) j\n\
                 WHERE n.node_id = ?1\n\
                 UNION ALL\n\
                 SELECT child.source_type, CAST(j.value AS INTEGER)\n\
                 FROM summary_nodes child\n\
                 JOIN source_walk walk\n\
                   ON walk.source_type = 'nodes'\n\
                  AND child.node_id = walk.source_id\n\
                 JOIN json_each(child.source_ids) j\n\
             )\n\
             SELECT 1\n\
             FROM source_walk walk\n\
             JOIN messages m\n\
               ON walk.source_type = 'messages'\n\
              AND m.store_id = walk.source_id\n\
             WHERE CASE\n\
                     WHEN ?2 = 'unknown' THEN (m.source = ?2 OR {blank})\n\
                     ELSE m.source = ?2\n\
                   END\n\
             LIMIT 1"
        );
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let matched = stmt.exists(params![node_id, normalized])?;
        cache.insert(node_id, matched);
        Ok(matched)
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
    /// `lcm_load_session`, honoring the optional `roles`/`time_from`/`time_to` filters
    /// (`load_session_page` + `_session_load_where`, `LCM:store.py:466-539`). Fetch `limit + 1` to
    /// detect `has_more` (§10.2).
    pub fn load_session_page(
        &self,
        session_id: &str,
        after_store_id: i64,
        limit: i64,
        roles: &[String],
        time_from: Option<f64>,
        time_to: Option<f64>,
    ) -> Result<Vec<MessageRow>> {
        let (where_sql, mut args) = session_load_where(session_id, roles, time_from, time_to);
        let sql = format!(
            "SELECT store_id, session_id, source, role, content, tool_call_id, tool_calls, \
             tool_name, timestamp, token_estimate FROM messages \
             WHERE {where_sql} AND store_id > ? ORDER BY store_id ASC LIMIT ?"
        );
        args.push(Value::Integer(after_store_id));
        args.push(Value::Integer(limit));
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_from_iter(args), map_message)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Total rows matching a `lcm_load_session` filter set (`count_session_load_messages`,
    /// `LCM:store.py:489-503`) — the response's `total_messages`.
    pub fn count_session_load_messages(
        &self,
        session_id: &str,
        roles: &[String],
        time_from: Option<f64>,
        time_to: Option<f64>,
    ) -> Result<i64> {
        let (where_sql, args) = session_load_where(session_id, roles, time_from, time_to);
        let sql = format!("SELECT COUNT(*) FROM messages WHERE {where_sql}");
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(&sql, params_from_iter(args), |r| r.get(0))?;
        Ok(n)
    }

    /// Sum of cached token estimates for a session (`get_session_token_total`,
    /// `LCM:store.py:578-584`) — `lcm_status`'s `store.estimated_tokens`.
    pub fn get_session_token_total(&self, session_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// The `MIN`/`MAX` `timestamp` over a set of `store_id`s — the real D0 time window
    /// (`get_time_bounds`, `LCM:store.py:657-667`). `(None, None)` for an empty set.
    pub fn get_time_bounds(&self, store_ids: &[i64]) -> Result<(Option<f64>, Option<f64>)> {
        if store_ids.is_empty() {
            return Ok((None, None));
        }
        let mut sql =
            String::from("SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE store_id IN (");
        let mut args: Vec<Value> = Vec::with_capacity(store_ids.len());
        for (i, id) in store_ids.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('?');
            args.push(Value::Integer(*id));
        }
        sql.push(')');
        let conn = self.conn.lock().expect("lcm store poisoned");
        let bounds = conn.query_row(&sql, params_from_iter(args), |r| {
            Ok((r.get::<_, Option<f64>>(0)?, r.get::<_, Option<f64>>(1)?))
        })?;
        Ok(bounds)
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

    /// The condensation feeder (§5.4): nodes at `depth` not referenced as a source by ANY
    /// deeper node (`get_uncondensed_at_depth`, `LCM:dag.py:309-327` uses `p.depth > depth`, not
    /// just `depth + 1` — a node carried into a skip-level condensation stays condensed), oldest
    /// first.
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
                 WHERE p.session_id = ?1 AND p.source_type = 'nodes' AND p.depth > ?2\
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

    /// Delete a session's nodes with `depth < min_depth` (`delete_below_depth`,
    /// `LCM:dag.py:233-246`) — the `/new` retain-depth prune (keep only high-level summaries).
    /// Returns the number of deleted nodes.
    pub fn delete_below_depth(&self, session_id: &str, min_depth: i64) -> Result<usize> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.execute(
            "DELETE FROM summary_nodes WHERE session_id = ?1 AND depth < ?2",
            params![session_id, min_depth],
        )?;
        Ok(n)
    }

    /// Delete all of a session's nodes (`delete_session_nodes`, `LCM:dag.py:248-256`). Returns the
    /// number of deleted nodes.
    pub fn delete_session_nodes(&self, session_id: &str) -> Result<usize> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.execute(
            "DELETE FROM summary_nodes WHERE session_id = ?1",
            [session_id],
        )?;
        Ok(n)
    }

    /// Move all nodes from one session to another (`reassign_session_nodes`,
    /// `LCM:dag.py:258-270`) — `/new` carry-over: retained summaries join the fresh session while
    /// node ids and node-to-node lineage stay intact. Returns the number of moved nodes.
    pub fn reassign_session_nodes(
        &self,
        old_session_id: &str,
        new_session_id: &str,
    ) -> Result<usize> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.execute(
            "UPDATE summary_nodes SET session_id = ?2 WHERE session_id = ?1",
            params![old_session_id, new_session_id],
        )?;
        Ok(n)
    }

    /// Move all persisted messages from one session to another (`reassign_session_messages`,
    /// `LCM:store.py:358-368`). Returns the number of moved rows.
    pub fn reassign_session_messages(
        &self,
        old_session_id: &str,
        new_session_id: &str,
    ) -> Result<usize> {
        if old_session_id.is_empty()
            || new_session_id.is_empty()
            || old_session_id == new_session_id
        {
            return Ok(0);
        }
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.execute(
            "UPDATE messages SET session_id = ?2 WHERE session_id = ?1",
            params![old_session_id, new_session_id],
        )?;
        Ok(n)
    }

    /// The deepest node depth recorded for `session_id` (`-1` when the session has no nodes) — the
    /// unlimited-condensation upper bound (`incremental_max_depth = -1`).
    pub fn max_depth(&self, session_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let d = conn.query_row(
            "SELECT COALESCE(MAX(depth), -1) FROM summary_nodes WHERE session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        Ok(d)
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

    /// Count a session's D0 nodes that reference at least one missing `messages` row — the Python
    /// `orphaned_dag_nodes` semantics (`LCM:tools.py:1819-1834`: per-session, `source_type =
    /// 'messages'`, a node counts once however many sources are gone).
    pub fn orphaned_session_node_count(&self, session_id: &str) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row(
            "SELECT COUNT(*) FROM summary_nodes p \
             WHERE p.session_id = ?1 AND p.source_type = 'messages' \
             AND EXISTS (SELECT 1 FROM json_each(p.source_ids) j \
                         WHERE NOT EXISTS (SELECT 1 FROM messages m \
                                           WHERE m.store_id = CAST(j.value AS INTEGER)))",
            [session_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// The session-scoped FTS coverage pair for `lcm_doctor`'s `fts_index_sync` check
    /// (`LCM:tools.py:1793-1817`): `(session messages, FTS rows joined to session messages)`.
    pub fn session_fts_sync_counts(&self, session_id: &str) -> Result<(i64, i64)> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let msg_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        let fts_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM messages_fts \
             JOIN messages ON messages_fts.rowid = messages.store_id \
             WHERE messages.session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        Ok((msg_count, fts_count))
    }

    // ---- payload-risk scan rows (§10.6 `scan_sqlite_payload_risks`) --------------------------

    /// The `limit` rows with the largest `content`/`tool_calls` values
    /// (`LCM:ingest_protection.py:1071-1088`).
    pub fn largest_field_rows(&self, field: RiskField, limit: i64) -> Result<Vec<RiskRow>> {
        let col = field.column();
        self.risk_rows(
            &format!(
                "SELECT store_id, session_id, source, role, COALESCE(length({col}), 0) AS len, \
                 {col} FROM messages ORDER BY len DESC LIMIT ?1"
            ),
            limit,
        )
    }

    /// Candidate rows whose `field` looks like a base64 data URI (SQL pre-filter; the caller
    /// applies the conservative regex — `LCM:ingest_protection.py:1094-1119`).
    pub fn data_uri_candidate_rows(&self, field: RiskField, cap: i64) -> Result<Vec<RiskRow>> {
        let col = field.column();
        self.risk_rows(
            &format!(
                "SELECT store_id, session_id, source, role, COALESCE(length({col}), 0) AS len, \
                 {col} FROM messages WHERE lower({col}) GLOB '*data:*;base64,*' \
                 ORDER BY len DESC LIMIT ?1"
            ),
            cap,
        )
    }

    /// Rows with an oversized `content` **or** `tool_calls` (both values returned for the caller's
    /// base64-run classification — `LCM:ingest_protection.py:1121-1131`).
    pub fn long_payload_rows(&self, min_chars: i64, cap: i64) -> Result<Vec<PayloadFieldsRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_calls FROM messages \
             WHERE COALESCE(length(content), 0) >= ?1 OR COALESCE(length(tool_calls), 0) >= ?1 \
             ORDER BY MAX(COALESCE(length(content), 0), COALESCE(length(tool_calls), 0)) DESC \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![min_chars, cap], |row| {
                Ok(PayloadFieldsRow {
                    store_id: row.get(0)?,
                    session_id: row.get(1)?,
                    source: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    role: row.get(3)?,
                    content: row.get(4)?,
                    tool_calls: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Assistant rows already quarantined at the store boundary (this port's placeholder family;
    /// the Python scan matches its own `quarantined_assistant_output` marker,
    /// `LCM:ingest_protection.py:1152-1165`).
    pub fn quarantined_assistant_rows(&self, limit: i64) -> Result<Vec<RiskRow>> {
        self.risk_rows(
            "SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS len, \
             content FROM messages \
             WHERE role = 'assistant' \
             AND content LIKE '[Externalized quarantined assistant output:%' \
             ORDER BY store_id DESC LIMIT ?1",
            limit,
        )
    }

    /// Large not-yet-quarantined assistant rows, biggest first, for the repetition classifier
    /// (`LCM:ingest_protection.py:1167-1178`).
    pub fn repetitive_assistant_candidate_rows(
        &self,
        min_chars: i64,
        cap: i64,
    ) -> Result<Vec<RiskRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS len, \
             content FROM messages \
             WHERE role = 'assistant' AND COALESCE(length(content), 0) >= ?1 \
             AND content NOT LIKE '[Externalized quarantined assistant output:%' \
             ORDER BY len DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![min_chars, cap], map_risk_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Short status-chatter candidates for the heartbeat classifier
    /// (`LCM:ingest_protection.py:1190-1210`).
    pub fn heartbeat_candidate_rows(&self, max_chars: i64, cap: i64) -> Result<Vec<RiskRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS len, \
             content FROM messages \
             WHERE role IN ('assistant', 'tool', 'system') \
             AND COALESCE(length(content), 0) BETWEEN 1 AND ?1 \
             AND (lower(trim(content)) GLOB 'still working*' \
               OR lower(trim(content)) GLOB 'working on it*' \
               OR lower(trim(content)) GLOB 'processing*' \
               OR lower(trim(content)) GLOB 'checking*' \
               OR lower(trim(content)) GLOB 'one moment*' \
               OR lower(trim(content)) GLOB 'ping*' \
               OR lower(trim(content)) GLOB 'heartbeat*' \
               OR lower(trim(content)) GLOB 'no update*') \
             ORDER BY store_id ASC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![max_chars, cap], map_risk_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Every row that carries an externalized-payload placeholder in `content` or `tool_calls`
    /// (`scan_externalized_payload_integrity`, `LCM:ingest_protection.py:1000-1008`), oldest first.
    pub fn externalized_ref_rows(&self) -> Result<Vec<PayloadFieldsRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(
            "SELECT store_id, session_id, source, role, content, tool_calls FROM messages \
             WHERE COALESCE(content, '') LIKE '%ref=%]%' \
                OR COALESCE(tool_calls, '') LIKE '%ref=%]%' \
             ORDER BY store_id ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(PayloadFieldsRow {
                    store_id: row.get(0)?,
                    session_id: row.get(1)?,
                    source: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    role: row.get(3)?,
                    content: row.get(4)?,
                    tool_calls: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Shared row mapper for the single-field risk queries above (one bound integer parameter).
    fn risk_rows(&self, sql: &str, bound: i64) -> Result<Vec<RiskRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params![bound], map_risk_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
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
        let blank = legacy_blank_source_clause("source");
        let sql = format!(
            "SELECT COUNT(*), \
             COALESCE(SUM(CASE WHEN source = 'unknown' THEN 1 ELSE 0 END), 0), \
             COALESCE(SUM(CASE WHEN {blank} THEN 1 ELSE 0 END), 0), \
             COALESCE(SUM(CASE WHEN NOT {blank} \
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
        let (
            messages_total,
            normalized_unknown_messages,
            legacy_blank_source_messages,
            attributed_messages,
        ) = match session_id {
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
        let message_sessions =
            session_set("SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL")?;
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
        let referenced: HashSet<String> = lifecycle_current
            .union(&lifecycle_finalized)
            .cloned()
            .collect();
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
            lifecycle_last_finalized_missing_in_messages: diff(
                &lifecycle_finalized,
                &message_sessions,
            ),
            lifecycle_last_finalized_missing_in_nodes: diff(&lifecycle_finalized, &node_sessions),
            lifecycle_last_finalized_missing_in_lcm_any: diff(&lifecycle_finalized, &lcm_any),
            message_sessions_without_lifecycle_current: diff(&message_sessions, &lifecycle_current),
            message_sessions_without_lifecycle_reference: diff(&message_sessions, &referenced),
            node_sessions_without_lifecycle_reference: diff(&node_sessions, &referenced),
        })
    }

    // ---- LifecycleStateStore (§4.5) --------------------------------------------------------

    /// The full lifecycle row for a conversation, if bound (`get_by_conversation`,
    /// `LCM:lifecycle_state.py:99`).
    pub fn get_lifecycle(&self, conversation_id: &str) -> Result<Option<LifecycleRow>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        get_lifecycle_locked(&conn, conversation_id)
    }

    /// Rows in `lcm_lifecycle_state` (`row_count`, `LCM:lifecycle_state.py:75`) — the empty-row GC
    /// threshold probe.
    pub fn lifecycle_row_count(&self) -> Result<i64> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let n = conn.query_row("SELECT COUNT(*) FROM lcm_lifecycle_state", [], |r| r.get(0))?;
        Ok(n)
    }

    /// Bind the active session for a conversation (`bind_session`,
    /// `LCM:lifecycle_state.py:123-227`). Re-binding the already-current session is a no-op;
    /// binding a *different* session resets the active frontier to 0, stamps `current_bound_at`,
    /// preserves the finalized/debt fields, and records `last_rollover_at` when the conversation
    /// visibly switched sessions. Returns the row as bound.
    pub fn bind_session(
        &self,
        conversation_id: &str,
        session_id: &str,
        now: f64,
    ) -> Result<LifecycleRow> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let existing = get_lifecycle_locked(&conn, conversation_id)?;
        if let Some(row) = existing {
            if row.current_session_id.as_deref() == Some(session_id) {
                return Ok(row);
            }
            // A different session takes over this conversation: fresh frontier + bound-at, keep
            // the finalized markers and debt, stamp the rollover when the switch is visible.
            let switched = row
                .current_session_id
                .as_deref()
                .is_some_and(|cur| cur != session_id)
                || (row.current_session_id.is_none()
                    && row
                        .last_finalized_session_id
                        .as_deref()
                        .is_some_and(|fin| fin != session_id));
            let last_rollover_at = if switched {
                Some(now)
            } else {
                row.last_rollover_at
            };
            conn.execute(
                "UPDATE lcm_lifecycle_state SET current_session_id = ?2, \
                     current_frontier_store_id = 0, current_bound_at = ?3, \
                     last_rollover_at = ?4, updated_at = ?3 \
                 WHERE conversation_id = ?1",
                params![conversation_id, session_id, now, last_rollover_at],
            )?;
        } else {
            conn.execute(
                "INSERT INTO lcm_lifecycle_state (conversation_id, current_session_id, \
                     current_bound_at, updated_at) VALUES (?1, ?2, ?3, ?3)",
                params![conversation_id, session_id, now],
            )?;
        }
        Ok(get_lifecycle_locked(&conn, conversation_id)?.expect("row just written"))
    }

    /// Finalize a session for its conversation (`finalize_session`,
    /// `LCM:lifecycle_state.py:229-272`): clear `current_session_id` when it matches, record the
    /// session as last-finalized, and advance the finalized frontier high-water mark. No-op when
    /// the conversation has no lifecycle row.
    pub fn finalize_session(
        &self,
        conversation_id: &str,
        session_id: &str,
        frontier_store_id: i64,
        now: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let Some(row) = get_lifecycle_locked(&conn, conversation_id)? else {
            return Ok(());
        };
        let (current_session, current_frontier) =
            if row.current_session_id.as_deref() == Some(session_id) {
                (None, 0i64)
            } else {
                (
                    row.current_session_id.clone(),
                    row.current_frontier_store_id,
                )
            };
        let finalized_frontier = frontier_store_id.max(row.last_finalized_frontier_store_id);
        conn.execute(
            "UPDATE lcm_lifecycle_state SET current_session_id = ?2, \
                 last_finalized_session_id = ?3, current_frontier_store_id = ?4, \
                 last_finalized_frontier_store_id = ?5, last_finalized_at = ?6, updated_at = ?6 \
             WHERE conversation_id = ?1",
            params![
                conversation_id,
                current_session,
                session_id,
                current_frontier,
                finalized_frontier,
                now
            ],
        )?;
        Ok(())
    }

    /// Record a completed old-session -> new-session rollover in one write (`record_rollover`,
    /// `LCM:lifecycle_state.py:274-335`): binds the new session with a zeroed active frontier,
    /// marks the old one finalized, and stamps every lifecycle timestamp. Idempotent when the row
    /// already reflects exactly this rollover.
    pub fn record_rollover(
        &self,
        conversation_id: &str,
        old_session_id: &str,
        new_session_id: &str,
        finalized_frontier_store_id: i64,
        now: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let existing = get_lifecycle_locked(&conn, conversation_id)?;
        if let Some(row) = &existing {
            if row.current_session_id.as_deref() == Some(new_session_id)
                && row.last_finalized_session_id.as_deref() == Some(old_session_id)
            {
                return Ok(());
            }
        }
        let finalized_frontier = finalized_frontier_store_id
            .max(existing.map_or(0, |r| r.last_finalized_frontier_store_id));
        conn.execute(
            "INSERT INTO lcm_lifecycle_state (conversation_id, current_session_id, \
                 last_finalized_session_id, current_frontier_store_id, \
                 last_finalized_frontier_store_id, current_bound_at, last_finalized_at, \
                 last_rollover_at, last_reset_at, updated_at) \
             VALUES (?1, ?2, ?3, 0, ?4, ?5, ?5, ?5, ?5, ?5) \
             ON CONFLICT(conversation_id) DO UPDATE SET \
                 current_session_id = excluded.current_session_id, \
                 last_finalized_session_id = excluded.last_finalized_session_id, \
                 current_frontier_store_id = 0, \
                 last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id, \
                 current_bound_at = excluded.current_bound_at, \
                 last_finalized_at = excluded.last_finalized_at, \
                 last_rollover_at = excluded.last_rollover_at, \
                 last_reset_at = excluded.last_reset_at, \
                 updated_at = excluded.updated_at",
            params![
                conversation_id,
                new_session_id,
                old_session_id,
                finalized_frontier,
                now
            ],
        )?;
        Ok(())
    }

    /// Stamp a `/new`-style reset: `last_reset_at` + cleared debt (`record_reset`,
    /// `LCM:lifecycle_state.py:616-636`). No-op when the conversation has no lifecycle row.
    pub fn record_reset(&self, conversation_id: &str, now: f64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "UPDATE lcm_lifecycle_state SET last_reset_at = ?2, debt_kind = NULL, \
                 debt_size_estimate = 0, debt_updated_at = ?2, updated_at = ?2 \
             WHERE conversation_id = ?1",
            params![conversation_id, now],
        )?;
        Ok(())
    }

    /// Stamp a maintenance attempt (`record_maintenance_attempt`,
    /// `LCM:lifecycle_state.py:597-614`). No-op when the conversation has no lifecycle row.
    pub fn record_maintenance_attempt(&self, conversation_id: &str, now: f64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "UPDATE lcm_lifecycle_state SET last_maintenance_attempt_at = ?2, updated_at = ?2 \
             WHERE conversation_id = ?1",
            params![conversation_id, now],
        )?;
        Ok(())
    }

    /// Delete lifecycle rows whose referenced sessions have no stored data (`prune_empty_sessions`,
    /// `LCM:lifecycle_state.py:638-753`): a row is eligible when BOTH its `current_session_id` and
    /// `last_finalized_session_id` have zero `messages` AND zero `summary_nodes`. `protected`
    /// sessions are never deleted; with `max_age_hours` set, rows younger than that (by bound-at,
    /// else finalized-at, else updated-at) are kept. Only the lifecycle table is touched. Returns
    /// the number of rows deleted.
    pub fn prune_empty_sessions(
        &self,
        protected: &[String],
        max_age_hours: Option<f64>,
        now: f64,
    ) -> Result<usize> {
        let mut conn = self.conn.lock().expect("lcm store poisoned");
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut deleted = 0usize;
        {
            let mut with_data: HashSet<String> = HashSet::new();
            for sql in [
                "SELECT DISTINCT session_id FROM messages",
                "SELECT DISTINCT session_id FROM summary_nodes",
            ] {
                let mut stmt = tx.prepare(sql)?;
                for s in stmt.query_map([], |r| r.get::<_, String>(0))? {
                    with_data.insert(s?);
                }
            }
            let max_age_seconds = max_age_hours.map(|h| h * 3600.0);
            let rows: Vec<(String, String, String, Option<f64>)> = {
                let mut stmt = tx.prepare(
                    "SELECT conversation_id, COALESCE(current_session_id, ''), \
                         COALESCE(last_finalized_session_id, ''), \
                         COALESCE(current_bound_at, last_finalized_at, updated_at) \
                     FROM lcm_lifecycle_state",
                )?;
                let mapped =
                    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
                mapped.collect::<std::result::Result<_, _>>()?
            };
            for (conversation_id, cur, fin, row_age) in rows {
                if (!cur.is_empty() && with_data.contains(&cur))
                    || (!fin.is_empty() && with_data.contains(&fin))
                {
                    continue;
                }
                if protected.iter().any(|p| p == &cur || p == &fin) {
                    continue;
                }
                if let (Some(max_age), Some(age_at)) = (max_age_seconds, row_age) {
                    if now - age_at < max_age {
                        continue;
                    }
                }
                tx.execute(
                    "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?1",
                    [&conversation_id],
                )?;
                deleted += 1;
            }
        }
        if deleted > 0 {
            tx.commit()?;
        }
        Ok(deleted)
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

    /// Clear a conversation's deferred compaction debt (`clear_debt`,
    /// `LCM:lifecycle_state.py:576-596`). No-op when the conversation has no lifecycle row.
    pub fn clear_debt(&self, conversation_id: &str, now: f64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        conn.execute(
            "UPDATE lcm_lifecycle_state SET debt_kind = NULL, debt_size_estimate = 0, \
                 debt_updated_at = ?2, updated_at = ?2 \
             WHERE conversation_id = ?1",
            params![conversation_id, now],
        )?;
        Ok(())
    }

    // ---- FTS hygiene (§4.4) ------------------------------------------------------------------

    /// Verify (and rebuild if broken) both external-content FTS indexes
    /// (`repair_external_content_fts`, `LCM:db_bootstrap.py:524-569`). `force` skips the deep
    /// integrity-check throttle (the explicit `/lcm doctor repair apply` path); the open path runs
    /// the throttled variant automatically.
    pub fn repair_fts(&self, force: bool) -> Result<Vec<FtsRepair>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let now = unix_now();
        let throttle = if force {
            None
        } else {
            Some(self.fts_check_interval_hours)
        };
        let mut out = Vec::with_capacity(2);
        for spec in [&schema::MESSAGES_FTS, &schema::NODES_FTS] {
            out.push(repair_fts_locked(&conn, spec, now, throttle)?);
        }
        // A repair pass is authoritative for the low-disk degradation flag: a degraded index
        // enters (or stays in) LIKE-only mode; a pass with every index verified/rebuilt cleanly
        // re-enables FTS (recovery after disk space is freed).
        self.degraded
            .store(out.iter().any(|o| o.degraded), Ordering::Relaxed);
        Ok(out)
    }

    /// The FTS5 deep integrity-check of both indexes, without repairing anything
    /// (`check_external_content_fts_integrity`, `LCM:db_bootstrap.py:424-463`) — `lcm_doctor`'s
    /// `fts_health` evidence.
    pub fn fts_integrity(&self) -> Result<Vec<FtsIntegrity>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        Ok(vec![
            check_fts_integrity_locked(&conn, &schema::MESSAGES_FTS),
            check_fts_integrity_locked(&conn, &schema::NODES_FTS),
        ])
    }

    /// The read-only `/lcm doctor repair` scan (`_scan_fts_repair`, `LCM:command.py:665-701`):
    /// per-index repair verdict (structural + unthrottled deep check) plus content/FTS row counts.
    /// Repairs nothing.
    pub fn scan_fts_repair(&self) -> Vec<FtsRepairScan> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let now = unix_now();
        [&schema::MESSAGES_FTS, &schema::NODES_FTS]
            .into_iter()
            .map(|spec| {
                let needs_repair = fts_needs_rebuild(&conn, spec, now, None);
                let count = |table: &str| -> Option<i64> {
                    conn.query_row(&format!("SELECT COUNT(*) FROM \"{table}\""), [], |r| {
                        r.get(0)
                    })
                    .ok()
                };
                FtsRepairScan {
                    table: spec.table.to_string(),
                    needs_repair,
                    content_rows: count(spec.content_table),
                    fts_rows: count(spec.table),
                }
            })
            .collect()
    }

    // ---- Operator maintenance (`/lcm backup` / `/lcm rotate`) ---------------------------------

    /// Copy the live database into `dest` via SQLite's online backup API (`_backup_database`,
    /// `LCM:command.py:454-489`). The connection autocommits (no pending-transaction flush is
    /// needed) and the backup API snapshots consistently under WAL.
    pub fn backup_to(&self, dest: &Path) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let mut dst = Connection::open(dest)?;
        let backup = rusqlite::backup::Backup::new(&conn, &mut dst)?;
        backup.run_to_completion(64, std::time::Duration::from_millis(50), None)?;
        Ok(())
    }

    /// The smallest `store_id` among the newest `tail` messages of a session — the rotate
    /// boundary probe (`get_session_tail`, `LCM:engine.py:4542`). `None` when the session has no
    /// rows.
    pub fn tail_min_store_id(&self, session_id: &str, tail: i64) -> Result<Option<i64>> {
        let conn = self.conn.lock().expect("lcm store poisoned");
        let v: Option<i64> = conn.query_row(
            "SELECT MIN(store_id) FROM (SELECT store_id FROM messages \
                 WHERE session_id = ?1 ORDER BY store_id DESC LIMIT ?2)",
            params![session_id, tail],
            |r| r.get(0),
        )?;
        Ok(v)
    }
}

impl Drop for Store {
    /// Graceful-shutdown hygiene (`Store.close`, `LCM:store.py:1018-1029`): checkpoint committed
    /// WAL frames into the main database file before releasing the connection. Best-effort — it
    /// doesn't run on crash/kill, and PASSIVE can leave frames behind while a reader is active.
    fn drop(&mut self) {
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()));
        }
    }
}

/// Round to one decimal place (matching the Python diagnostics' `round(x, 1)`).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Read one lifecycle row while already holding the connection lock.
fn get_lifecycle_locked(conn: &Connection, conversation_id: &str) -> Result<Option<LifecycleRow>> {
    let row = conn
        .query_row(
            "SELECT conversation_id, current_session_id, last_finalized_session_id, \
                 current_frontier_store_id, last_finalized_frontier_store_id, debt_kind, \
                 debt_size_estimate, current_bound_at, last_finalized_at, debt_updated_at, \
                 last_maintenance_attempt_at, last_rollover_at, last_reset_at, updated_at \
             FROM lcm_lifecycle_state WHERE conversation_id = ?1",
            [conversation_id],
            |r| {
                Ok(LifecycleRow {
                    conversation_id: r.get(0)?,
                    current_session_id: r.get(1)?,
                    last_finalized_session_id: r.get(2)?,
                    current_frontier_store_id: r.get(3)?,
                    last_finalized_frontier_store_id: r.get(4)?,
                    debt_kind: r.get(5)?,
                    debt_size_estimate: r.get(6)?,
                    current_bound_at: r.get(7)?,
                    last_finalized_at: r.get(8)?,
                    debt_updated_at: r.get(9)?,
                    last_maintenance_attempt_at: r.get(10)?,
                    last_rollover_at: r.get(11)?,
                    last_reset_at: r.get(12)?,
                    updated_at: r.get(13)?,
                })
            },
        )
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(other),
        })?;
    Ok(row)
}

// ---- External-content FTS hygiene helpers (`LCM:db_bootstrap.py`) -------------------------------
//
// All table/trigger identifiers below come from the compile-time `FtsSpec` constants (never user
// input), so direct interpolation is safe (the Python `quote_sql_identifier` guard has no runtime
// counterpart to defend against).

/// Unix now (fractional seconds) for the open-path repair, which has no caller-supplied clock.
fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The FTS5 shadow tables backing an external-content index.
fn fts_shadow_names(spec: &schema::FtsSpec) -> [String; 4] {
    ["data", "idx", "docsize", "config"].map(|s| format!("{}_{s}", spec.table))
}

/// The metadata key throttling a spec's deep integrity-check.
fn fts_marker_key(spec: &schema::FtsSpec) -> String {
    format!("fts_integrity_checked_at:{}", spec.table)
}

/// When the deep check last passed for this spec, if recorded.
fn load_fts_checked_at(conn: &Connection, spec: &schema::FtsSpec) -> Option<f64> {
    conn.query_row(
        "SELECT value FROM metadata WHERE key = ?1",
        [fts_marker_key(spec)],
        |r| r.get::<_, String>(0),
    )
    .ok()?
    .parse()
    .ok()
}

/// Stamp the deep-check marker (`_record_integrity_checked`, `LCM:db_bootstrap.py:371-383`).
fn record_fts_checked(conn: &Connection, spec: &schema::FtsSpec, now: f64) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?1, ?2) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![fts_marker_key(spec), now.to_string()],
    )?;
    Ok(())
}

/// Whether the throttled open path should run the deep check now
/// (`_should_run_integrity_check`, `LCM:db_bootstrap.py:386-398`): `0` hours checks every time,
/// negative never checks, non-finite falls back to the default.
fn should_run_fts_integrity(
    conn: &Connection,
    spec: &schema::FtsSpec,
    now: f64,
    hours: f64,
) -> bool {
    let hours = if hours.is_finite() {
        hours
    } else {
        DEFAULT_FTS_CHECK_INTERVAL_HOURS
    };
    if hours == 0.0 {
        return true;
    }
    if hours < 0.0 {
        return false;
    }
    match load_fts_checked_at(conn, spec) {
        None => true,
        Some(last) => now - last >= hours * 3600.0,
    }
}

/// Cheap structural health of an external-content index (`_fts_needs_rebuild_structural`,
/// `LCM:db_bootstrap.py:282-324`): the virtual table + all four shadows exist, the DDL is an FTS5
/// virtual table exposing the indexed column, and the `_docsize` shadow's row count (the true
/// indexed-document count — `COUNT(*)` on the FTS table reads through to content) matches the
/// content table. Any SQLite error counts as "needs rebuild".
fn fts_needs_rebuild_structural(conn: &Connection, spec: &schema::FtsSpec) -> bool {
    let check = || -> rusqlite::Result<bool> {
        let shadows = fts_shadow_names(spec);
        let mut existing: HashSet<String> = HashSet::new();
        {
            let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type = 'table'")?;
            for name in stmt.query_map([], |r| r.get::<_, String>(0))? {
                existing.insert(name?);
            }
        }
        if !existing.contains(spec.table) || shadows.iter().any(|s| !existing.contains(s)) {
            return Ok(true);
        }
        let ddl: Option<String> = conn.query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [spec.table],
            |r| r.get(0),
        )?;
        let normalized = ddl.unwrap_or_default().to_lowercase();
        if !normalized.contains("virtual table") || !normalized.contains("using fts5") {
            return Ok(true);
        }
        let mut has_indexed_column = false;
        {
            let mut stmt = conn.prepare(&format!("PRAGMA table_info(\"{}\")", spec.table))?;
            for name in stmt.query_map([], |r| r.get::<_, String>(1))? {
                if name? == spec.indexed_column {
                    has_indexed_column = true;
                    break;
                }
            }
        }
        if !has_indexed_column {
            return Ok(true);
        }
        let content: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM \"{}\"", spec.content_table),
            [],
            |r| r.get(0),
        )?;
        let docsize: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM \"{}_docsize\"", spec.table),
            [],
            |r| r.get(0),
        )?;
        Ok(content != docsize)
    };
    check().unwrap_or(true)
}

/// FTS5's own deep integrity-check (`check_external_content_fts_integrity`,
/// `LCM:db_bootstrap.py:424-463`), exposed as a special `INSERT`; `rank = 1` verifies the index
/// against the external content table. Wrapped in a savepoint and rolled back so diagnostics
/// leave no state behind on the shared connection.
fn check_fts_integrity_locked(conn: &Connection, spec: &schema::FtsSpec) -> FtsIntegrity {
    let verdict = |status: &str, detail: String| FtsIntegrity {
        table: spec.table.to_string(),
        status: status.to_string(),
        detail,
    };
    if fts_needs_rebuild_structural(conn, spec) {
        return verdict("fail", "structural repair needed".to_string());
    }
    let savepoint = format!("\"lcm_fts_integrity_{}\"", spec.table);
    if let Err(e) = conn.execute_batch(&format!("SAVEPOINT {savepoint};")) {
        return verdict("fail", e.to_string());
    }
    let insert = format!(
        "INSERT INTO \"{t}\"(\"{t}\", rank) VALUES('integrity-check', 1)",
        t = spec.table
    );
    match conn.execute(&insert, []) {
        Ok(_) => {
            match conn.execute_batch(&format!("ROLLBACK TO {savepoint}; RELEASE {savepoint};")) {
                Ok(()) => verdict("pass", "ok".to_string()),
                Err(e) => verdict("fail", e.to_string()),
            }
        }
        Err(e) => {
            let _ = conn.execute_batch(&format!("ROLLBACK TO {savepoint}; RELEASE {savepoint};"));
            let detail = e.to_string();
            let lowered = detail.to_lowercase();
            if lowered.contains("readonly") || lowered.contains("read-only") {
                verdict("unchecked", detail)
            } else {
                verdict("fail", detail)
            }
        }
    }
}

/// Structural check first, then the (possibly throttled) deep check (`_fts_needs_rebuild`,
/// `LCM:db_bootstrap.py:401-421`). The deep check is O(index size) and was the dominant startup
/// cost on large banks, hence the marker throttle on the open path; explicit repair passes
/// `throttle = None` so it can catch same-row-count drift the structural check cannot see.
fn fts_needs_rebuild(
    conn: &Connection,
    spec: &schema::FtsSpec,
    now: f64,
    throttle: Option<f64>,
) -> bool {
    if fts_needs_rebuild_structural(conn, spec) {
        return true;
    }
    if let Some(hours) = throttle {
        if !should_run_fts_integrity(conn, spec, now, hours) {
            return false;
        }
    }
    let result = check_fts_integrity_locked(conn, spec);
    if result.status == "pass" {
        let _ = record_fts_checked(conn, spec, now);
    }
    result.status == "fail"
}

/// The trigger name out of our fixed `CREATE TRIGGER IF NOT EXISTS <name> …` DDL shape.
fn fts_trigger_name(trigger_sql: &str) -> Option<&str> {
    trigger_sql.split_whitespace().nth(5)
}

/// Whether any of the spec's sync triggers is missing (`_fts_missing_triggers`,
/// `LCM:db_bootstrap.py:503-517`).
fn fts_missing_triggers(conn: &Connection, spec: &schema::FtsSpec) -> rusqlite::Result<bool> {
    for sql in spec.trigger_sqls {
        let Some(name) = fts_trigger_name(sql) else {
            continue;
        };
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?1)",
            [name],
            |r| r.get(0),
        )?;
        if exists == 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Rebuild a broken index from its content table and recreate any missing sync trigger
/// (`repair_external_content_fts`, `LCM:db_bootstrap.py:524-569`).
///
/// Low-disk degradation (`LCM:db_bootstrap.py:533-545`): a rebuild that hits a disk-space-class
/// write error drops the index's artifacts (triggers included, so base-table writes keep working)
/// and reports `degraded` instead of failing — the store falls back to LIKE-only search. Python
/// probes free space up front (`_check_disk_space`, statvfs); here the *write error itself* is the
/// probe, so degradation triggers on the actual failure with no platform dependency.
fn repair_fts_locked(
    conn: &Connection,
    spec: &schema::FtsSpec,
    now: f64,
    throttle: Option<f64>,
) -> Result<FtsRepair> {
    let mut rebuilt = false;
    if fts_needs_rebuild(conn, spec, now, throttle) {
        if let Err(e) = rebuild_fts(conn, spec) {
            if is_disk_full(&e) {
                drop_fts_artifacts(conn, spec);
                tracing::warn!(
                    table = spec.table,
                    error = %e,
                    "lcm: FTS rebuild hit a full disk; degrading to LIKE-only search"
                );
                return Ok(FtsRepair {
                    table: spec.table.to_string(),
                    rebuilt: false,
                    triggers_recreated: false,
                    degraded: true,
                });
            }
            return Err(e.into());
        }
        tracing::warn!(
            table = spec.table,
            "lcm: rebuilt broken FTS index from content table"
        );
        rebuilt = true;
    }
    let triggers_were_missing = fts_missing_triggers(conn, spec)?;
    for sql in spec.trigger_sqls {
        conn.execute_batch(sql)?;
    }
    if rebuilt {
        // A freshly rebuilt index is known-consistent; let the next open skip the deep check.
        record_fts_checked(conn, spec, now)?;
    }
    Ok(FtsRepair {
        table: spec.table.to_string(),
        rebuilt,
        triggers_recreated: triggers_were_missing,
        degraded: false,
    })
}

/// The write half of the repair: drop the virtual table (which owns its shadows), sweep orphaned
/// shadows left by a half-broken index, recreate with the byte-identical schema DDL, and reindex
/// from the content table.
fn rebuild_fts(conn: &Connection, spec: &schema::FtsSpec) -> rusqlite::Result<()> {
    conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{}\";", spec.table))?;
    for shadow in fts_shadow_names(spec) {
        conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{shadow}\";"))?;
    }
    conn.execute_batch(spec.create_sql)?;
    conn.execute(
        &format!(
            "INSERT INTO \"{t}\"(\"{t}\") VALUES('rebuild')",
            t = spec.table
        ),
        [],
    )?;
    Ok(())
}

/// Whether a SQLite failure is a disk-space-class write error (`SQLITE_FULL` / `SQLITE_IOERR`) —
/// the class where degrading to LIKE-only search beats failing the open.
fn is_disk_full(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(f, _)
            if matches!(
                f.code,
                rusqlite::ErrorCode::DiskFull | rusqlite::ErrorCode::SystemIoFailure
            )
    )
}

/// Best-effort removal of one index's FTS artifacts — sync triggers first (so base-table writes
/// stop touching the missing index), then the virtual table + orphaned shadows
/// (`_drop_fts_artifacts`, `LCM:db_bootstrap.py:466-492`). Errors are ignored: this runs on an
/// already-degraded (full) disk and drops only free pages.
fn drop_fts_artifacts(conn: &Connection, spec: &schema::FtsSpec) {
    for sql in spec.trigger_sqls {
        if let Some(name) = fts_trigger_name(sql) {
            let _ = conn.execute_batch(&format!("DROP TRIGGER IF EXISTS \"{name}\";"));
        }
    }
    let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{}\";", spec.table));
    for shadow in fts_shadow_names(spec) {
        let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{shadow}\";"));
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
    // Read-side source normalization (`_row_to_dict`, `LCM:store.py:994`): legacy NULL/blank rows
    // surface as `unknown` without rewriting the stored value.
    let source: Option<String> = row.get(2)?;
    let source = match source.as_deref().map(str::trim) {
        None | Some("") => "unknown".to_string(),
        Some(s) => s.to_string(),
    };
    Ok(MessageRow {
        store_id: row.get(0)?,
        session_id: row.get(1)?,
        source,
        role: row.get(3)?,
        content: row.get(4)?,
        tool_call_id: row.get(5)?,
        tool_calls: row.get(6)?,
        tool_name: row.get(7)?,
        timestamp: row.get(8)?,
        token_estimate: row.get(9)?,
    })
}

/// SQL matching a legacy NULL/blank `source` (`_legacy_blank_source_clause`, `LCM:store.py:61-66`).
/// SQLite `TRIM()` strips only spaces by default; the explicit character set mirrors the
/// write-time `str.trim()` so legacy tabs/newlines don't form a fake attributed-source bucket.
fn legacy_blank_source_clause(column: &str) -> String {
    const WS: &str = "char(9) || char(10) || char(11) || char(12) || char(13) || char(32)";
    format!("({column} IS NULL OR TRIM({column}, {WS}) = '')")
}

/// Which message field a payload-risk query samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiskField {
    /// The `content` column.
    Content,
    /// The `tool_calls` column.
    ToolCalls,
}

impl RiskField {
    /// The backing column name (a compile-time literal, never interpolated user input).
    pub fn column(self) -> &'static str {
        match self {
            RiskField::Content => "content",
            RiskField::ToolCalls => "tool_calls",
        }
    }

    /// The Python detail-field label (`"content"` / `"tool_calls"`).
    pub fn label(self) -> &'static str {
        self.column()
    }
}

/// One sampled row of the §10.6 payload-risk scan: row metadata, the sampled field's length, and
/// its raw value (kept in-crate for classification; the doctor detail carries metadata only).
#[derive(Clone, Debug)]
pub struct RiskRow {
    /// Row id.
    pub store_id: i64,
    /// Owning session.
    pub session_id: String,
    /// The row's source bucket (raw column value; blank for legacy rows).
    pub source: String,
    /// The row's role.
    pub role: String,
    /// `length()` of the sampled field.
    pub length: i64,
    /// The sampled field value.
    pub value: Option<String>,
}

/// A row sampled with **both** payload-carrying fields (`content` + `tool_calls`).
#[derive(Clone, Debug)]
pub struct PayloadFieldsRow {
    /// Row id.
    pub store_id: i64,
    /// Owning session.
    pub session_id: String,
    /// The row's source bucket (raw column value; blank for legacy rows).
    pub source: String,
    /// The row's role.
    pub role: String,
    /// The `content` column.
    pub content: Option<String>,
    /// The `tool_calls` column.
    pub tool_calls: Option<String>,
}

/// Row mapper for the single-field risk queries (store_id, session, source, role, len, value).
fn map_risk_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RiskRow> {
    Ok(RiskRow {
        store_id: row.get(0)?,
        session_id: row.get(1)?,
        source: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
        role: row.get(3)?,
        length: row.get(4)?,
        value: row.get(5)?,
    })
}

/// The `lcm_load_session` WHERE clause + bind values (`_session_load_where`,
/// `LCM:store.py:466-486`): session scope plus the optional `role IN (…)` and inclusive
/// timestamp bounds.
fn session_load_where(
    session_id: &str,
    roles: &[String],
    time_from: Option<f64>,
    time_to: Option<f64>,
) -> (String, Vec<Value>) {
    let mut clauses = vec!["session_id = ?".to_string()];
    let mut args: Vec<Value> = vec![Value::Text(session_id.to_string())];
    if !roles.is_empty() {
        let placeholders = vec!["?"; roles.len()].join(",");
        clauses.push(format!("role IN ({placeholders})"));
        args.extend(roles.iter().map(|r| Value::Text(r.clone())));
    }
    if let Some(from) = time_from {
        clauses.push("timestamp >= ?".to_string());
        args.push(Value::Real(from));
    }
    if let Some(to) = time_to {
        clauses.push("timestamp <= ?".to_string());
        args.push(Value::Real(to));
    }
    (clauses.join(" AND "), args)
}

/// Append the optional `MessageFilter` clauses (and their bind values) to a search query whose
/// table alias is `m`. Filters use anonymous `?` placeholders bound in append order. Filtering by
/// `unknown` (or a blank source, which normalizes to it) also matches legacy NULL/blank rows
/// written before source normalization (`_source_filter_clause`, `LCM:store.py:74-80`).
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
        let normalized = match source.trim() {
            "" => "unknown",
            s => s,
        };
        if normalized == "unknown" {
            sql.push_str(&format!(
                " AND (m.source = ? OR {})",
                legacy_blank_source_clause("m.source")
            ));
        } else {
            sql.push_str(" AND m.source = ?");
        }
        args.push(Value::Text(normalized.to_string()));
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

/// Role bias for search ordering (`_MESSAGE_ROLE_BIAS_SQL`, `LCM:store.py:53`): user rows outrank
/// assistant rows outrank tool rows at equal relevance.
const MESSAGE_ROLE_BIAS_SQL: &str =
    "CASE m.role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END";

/// The message FTS `ORDER BY` per sort mode (`_build_search_order_by`, `LCM:store.py:102-126`,
/// with `timestamp_expr = m.timestamp` and the role-bias penalty).
fn message_search_order_by(sort: SortMode) -> String {
    match sort {
        SortMode::Relevance => {
            format!("rank ASC, {MESSAGE_ROLE_BIAS_SQL} ASC, m.timestamp DESC")
        }
        SortMode::Hybrid => format!(
            "(rank / (1 + (MAX(0.0, ((strftime('%s','now') - m.timestamp) / 3600.0)) * {AGE_DECAY_RATE}))) ASC, \
             {MESSAGE_ROLE_BIAS_SQL} ASC, m.timestamp DESC"
        ),
        SortMode::Recency => {
            format!("m.timestamp DESC, {MESSAGE_ROLE_BIAS_SQL} ASC, rank ASC")
        }
    }
}

/// The node FTS `ORDER BY` per sort mode (`_build_search_order_by`, `LCM:dag.py:51-60`, with
/// `recency_expr = COALESCE(n.latest_at, n.created_at)` and no role bias).
fn node_search_order_by(sort: SortMode) -> String {
    const RECENCY: &str = "COALESCE(n.latest_at, n.created_at)";
    match sort {
        SortMode::Relevance => format!("rank ASC, {RECENCY} DESC"),
        SortMode::Hybrid => format!(
            "(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {RECENCY}) / 3600.0)) * {AGE_DECAY_RATE}))) ASC, \
             {RECENCY} DESC"
        ),
        SortMode::Recency => format!("{RECENCY} DESC"),
    }
}

/// The shared LIKE-fallback WHERE clause (`LCM:store.py:805-827`): non-NULL content, the optional
/// message filters, and an OR of `content LIKE '%term%'` per term.
fn like_message_where(like_terms: &[String], filter: &MessageFilter<'_>) -> (String, Vec<Value>) {
    let mut sql = String::from("m.content IS NOT NULL");
    let mut args: Vec<Value> = Vec::new();
    push_message_filters(&mut sql, &mut args, filter);
    sql.push_str(" AND (");
    for (i, term) in like_terms.iter().enumerate() {
        if i > 0 {
            sql.push_str(" OR ");
        }
        sql.push_str("m.content LIKE ? ESCAPE '\\'");
        args.push(Value::Text(format!("%{}%", like_escape(term))));
    }
    sql.push(')');
    (sql, args)
}

/// SQL counting non-overlapping case-insensitive occurrences of a term in `content`
/// (`count_expr`, `LCM:store.py:838-843`). Binds the term twice.
const LIKE_COUNT_EXPR: &str = "((LENGTH(LOWER(m.content)) - \
     LENGTH(REPLACE(LOWER(m.content), LOWER(?), ''))) / NULLIF(LENGTH(?), 0))";

/// The recency-sort `ORDER BY` for the message LIKE fallback (`LCM:store.py:835-902`): newest
/// first, then role bias, SQL-side term-hit score (collapsed to 0/1 per term for risky-ASCII
/// queries), a SQL transliteration of the directness score, and `store_id` as the final tie-break.
fn like_recency_order_by(
    terms: &[String],
    phrases: &[String],
    collapse_risky_repeats: bool,
) -> (String, Vec<Value>) {
    let mut order_args: Vec<Value> = Vec::new();

    let mut score_exprs: Vec<String> = Vec::new();
    for term in terms {
        if collapse_risky_repeats {
            score_exprs.push("CASE WHEN m.content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END".into());
            order_args.push(Value::Text(format!("%{}%", like_escape(term))));
        } else {
            score_exprs.push(LIKE_COUNT_EXPR.into());
            order_args.push(Value::Text(term.clone()));
            order_args.push(Value::Text(term.clone()));
        }
    }
    let score_expr = if score_exprs.is_empty() {
        "0".to_string()
    } else {
        score_exprs.join(" + ")
    };

    let count_sum = |selected: &[&String], args: &mut Vec<Value>, unique: bool| -> String {
        let mut parts: Vec<String> = Vec::new();
        for term in selected {
            if unique {
                parts.push(format!(
                    "CASE WHEN ({LIKE_COUNT_EXPR}) > 0 THEN 1 ELSE 0 END"
                ));
            } else {
                parts.push(LIKE_COUNT_EXPR.into());
            }
            args.push(Value::Text((*term).clone()));
            args.push(Value::Text((*term).clone()));
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    };

    let mut directness_args: Vec<Value> = Vec::new();
    let all_terms: Vec<&String> = terms.iter().collect();
    let unique_score_expr = count_sum(&all_terms, &mut directness_args, true);
    let directness_expr = if phrases.is_empty() {
        let total_expr = count_sum(&all_terms, &mut directness_args, false);
        let unique_expr = count_sum(&all_terms, &mut directness_args, true);
        let repetition = format!("MAX(({total_expr}) - ({unique_expr}), 0)");
        format!("(({unique_score_expr}) * 5.0) - MIN(({repetition}), 6)")
    } else {
        let mut phrase_hit_exprs: Vec<String> = Vec::new();
        for phrase in phrases {
            phrase_hit_exprs
                .push("CASE WHEN INSTR(LOWER(m.content), LOWER(?)) > 0 THEN 1 ELSE 0 END".into());
            directness_args.push(Value::Text(phrase.clone()));
        }
        let phrase_hit_expr = phrase_hit_exprs.join(" + ");
        let normalized_phrases: HashSet<String> =
            phrases.iter().map(|p| p.trim().to_lowercase()).collect();
        let non_phrase_terms: Vec<&String> = terms
            .iter()
            .filter(|t| !normalized_phrases.contains(&t.trim().to_lowercase()))
            .collect();
        let total_expr = count_sum(&non_phrase_terms, &mut directness_args, false);
        let unique_expr = count_sum(&non_phrase_terms, &mut directness_args, true);
        let repetition = format!("MAX(({total_expr}) - ({unique_expr}), 0)");
        format!(
            "(({unique_score_expr}) * 5.0) + (({phrase_hit_expr}) * 8.0) - MIN(({repetition}), 6)"
        )
    };
    order_args.extend(directness_args);

    let order_by = format!(
        "ORDER BY m.timestamp DESC, {MESSAGE_ROLE_BIAS_SQL} ASC, ({score_expr}) DESC, \
         ({directness_expr}) DESC, m.store_id DESC"
    );
    (order_by, order_args)
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
    /// test -p daemon-context-lcm schema_matches_golden`.
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
        assert_eq!(
            store.get_uncondensed_at_depth("s1", 0, 10).unwrap().len(),
            4
        );

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
        assert!(store
            .get_uncondensed_at_depth("s1", 0, 10)
            .unwrap()
            .is_empty());
        let frontier = store.get_uncondensed_frontier("s1").unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].depth, 1);
        assert_eq!(frontier[0].source_type, SourceType::Nodes);
    }

    /// `get_uncondensed_at_depth` parity (`LCM:dag.py:309-327`): a node referenced by ANY deeper
    /// node counts as condensed — including a skip-level parent (depth 0 -> depth 2).
    #[test]
    fn uncondensed_excludes_skip_level_parents() {
        let store = Store::open_in_memory().expect("open");
        let node = |depth: i64, source_ids: Vec<i64>, source_type: SourceType| NewNode {
            session_id: "s1".into(),
            depth,
            summary: format!("d{depth}"),
            token_count: 2,
            source_token_count: 10,
            source_ids,
            source_type,
            created_at: depth as f64,
            earliest_at: None,
            latest_at: None,
            expand_hint: String::new(),
        };
        let leaf = store
            .add_node(&node(0, vec![1], SourceType::Messages))
            .unwrap();
        store
            .add_node(&node(2, vec![leaf], SourceType::Nodes))
            .unwrap();
        assert!(
            store
                .get_uncondensed_at_depth("s1", 0, 10)
                .unwrap()
                .is_empty(),
            "a skip-level (depth 2) parent still marks the leaf condensed"
        );
    }

    /// Transcript-GC guards (`gc_externalized_tool_result`, `LCM:store.py:381-405`): only unpinned
    /// tool rows are rewritten, the rewrite is idempotent, and the cached token estimate follows
    /// the placeholder.
    #[test]
    fn gc_rewrite_guards_role_pinned_and_reestimates_tokens() {
        let store = Store::open_in_memory().expect("open");
        let ext = "[Externalized tool output: tool_call_id=c1; chars=9000; bytes=9000; ref=r.json]";
        let ids = store
            .append_batch(
                "s1",
                &[
                    NewMessage {
                        role: "tool".into(),
                        content: Some(ext.into()),
                        tool_call_id: Some("c1".into()),
                        token_estimate: 2000,
                        ..Default::default()
                    },
                    NewMessage {
                        role: "user".into(),
                        content: Some(ext.into()),
                        token_estimate: 2000,
                        ..Default::default()
                    },
                    NewMessage {
                        role: "tool".into(),
                        content: Some(ext.into()),
                        tool_call_id: Some("c2".into()),
                        token_estimate: 2000,
                        ..Default::default()
                    },
                ],
                10.0,
            )
            .unwrap();
        let (tool_id, user_id, pinned_id) = (ids[0], ids[1], ids[2]);
        store
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE messages SET pinned = 1 WHERE store_id = ?1",
                [pinned_id],
            )
            .unwrap();

        // Candidates: unpinned tool rows only.
        let candidates = store.messages_to_gc("s1", i64::MAX).unwrap();
        assert_eq!(
            candidates.iter().map(|r| r.store_id).collect::<Vec<_>>(),
            vec![tool_id]
        );

        let placeholder = "[GC'd externalized tool output: ref=r.json]";
        assert!(!store
            .gc_externalized_tool_result(user_id, placeholder, 11)
            .unwrap());
        assert!(!store
            .gc_externalized_tool_result(pinned_id, placeholder, 11)
            .unwrap());
        assert!(store
            .gc_externalized_tool_result(tool_id, placeholder, 11)
            .unwrap());
        // Idempotent: the second pass sees the placeholder already in place.
        assert!(!store
            .gc_externalized_tool_result(tool_id, placeholder, 11)
            .unwrap());

        let row = store.get_message(tool_id).unwrap().unwrap();
        assert_eq!(row.content.as_deref(), Some(placeholder));
        assert_eq!(row.token_estimate, 11, "token estimate follows the rewrite");
        let untouched = store.get_message(pinned_id).unwrap().unwrap();
        assert_eq!(untouched.content.as_deref(), Some(ext));
    }

    /// `unknown`-source filtering matches legacy NULL/blank rows (`_source_filter_clause`,
    /// `LCM:store.py:74-80`), and write-side normalization trims whitespace-only sources.
    #[test]
    fn unknown_source_filter_matches_legacy_blank_rows() {
        let store = Store::open_in_memory().expect("open");
        let msg = |source: &str| NewMessage {
            source: source.into(),
            role: "user".into(),
            content: Some("hello lineage".into()),
            token_estimate: 2,
            ..Default::default()
        };
        let ids = store
            .append_batch("s1", &[msg("cli"), msg("unknown"), msg(" \t")], 10.0)
            .unwrap();
        assert_eq!(
            store.get_message(ids[2]).unwrap().unwrap().source,
            "unknown",
            "whitespace-only source normalizes to unknown on write"
        );
        // Legacy rows written before normalization: NULL and tab-padded blank.
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, source, role, content, timestamp) \
                 VALUES ('s1', NULL, 'user', 'hello lineage', 11.0)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, source, role, content, timestamp) \
                 VALUES ('s1', char(9), 'user', 'hello lineage', 12.0)",
                [],
            )
            .unwrap();
        }
        let filter = MessageFilter {
            source: Some("unknown"),
            ..Default::default()
        };
        let hits = store
            .search_messages_like_unordered(&["lineage".into()], &filter, 50)
            .unwrap();
        assert_eq!(
            hits.len(),
            4,
            "unknown matches the normalized row + both legacy blanks, not the attributed one"
        );
        assert!(hits.iter().all(|r| r.store_id != ids[0]));
        // A blank filter value normalizes to `unknown` too.
        let blank = MessageFilter {
            source: Some("  "),
            ..Default::default()
        };
        assert_eq!(
            store
                .search_messages_like_unordered(&["lineage".into()], &blank, 50)
                .unwrap()
                .len(),
            4
        );
    }

    /// A healthy bank passes the startup repair untouched, records the deep-check marker, and
    /// reports `pass` integrity.
    #[test]
    fn fts_repair_is_noop_on_healthy_store() {
        let store = Store::open_in_memory().expect("open");
        for outcome in store.repair_fts(true).unwrap() {
            assert!(
                !outcome.rebuilt,
                "{}: healthy index not rebuilt",
                outcome.table
            );
            assert!(!outcome.triggers_recreated);
        }
        for verdict in store.fts_integrity().unwrap() {
            assert_eq!(
                verdict.status, "pass",
                "{}: {}",
                verdict.table, verdict.detail
            );
        }
        let marker: String = store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT value FROM metadata WHERE key = 'fts_integrity_checked_at:messages_fts'",
                [],
                |r| r.get(0),
            )
            .expect("open-path deep check recorded its marker");
        assert!(marker.parse::<f64>().unwrap() > 0.0);
    }

    /// A desynced index (sync trigger dropped, rows written past it) is detected structurally and
    /// rebuilt from the content table on the next open, trigger included
    /// (`repair_external_content_fts`, `LCM:db_bootstrap.py:524-569`).
    #[test]
    fn fts_rebuild_heals_desynced_index_on_open() {
        let dir = std::env::temp_dir().join(format!("lcm-fts-heal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("lcm.db");
        {
            let store = Store::open(&path).expect("open");
            store
                .append_batch(
                    "s1",
                    &[NewMessage {
                        role: "user".into(),
                        content: Some("indexed before corruption".into()),
                        token_estimate: 3,
                        ..Default::default()
                    }],
                    10.0,
                )
                .unwrap();
            // Sabotage: drop the insert trigger, then write a row the index never sees.
            let conn = store.conn.lock().unwrap();
            conn.execute_batch("DROP TRIGGER msg_fts_insert;").unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, source, role, content, timestamp) \
                 VALUES ('s1', 'cli', 'user', 'written past the dropped trigger', 11.0)",
                [],
            )
            .unwrap();
        }
        let healed = Store::open(&path).expect("reopen heals");
        let counts = healed.table_counts().unwrap();
        assert_eq!(counts.messages, 2);
        assert_eq!(counts.messages_fts, 2, "rebuild reindexed the missed row");
        let hits = healed.search_messages("s1", "dropped trigger", 10).unwrap();
        assert_eq!(hits.len(), 1, "the missed row is searchable after the heal");
        // The trigger is back: new writes index again.
        healed
            .append_batch(
                "s1",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("post-heal write".into()),
                    token_estimate: 2,
                    ..Default::default()
                }],
                12.0,
            )
            .unwrap();
        assert_eq!(
            healed.search_messages("s1", "post heal", 10).unwrap().len(),
            1
        );
        drop(healed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Disk-space-class SQLite failures (and only those) select the degraded path.
    #[test]
    fn disk_full_error_classification() {
        let failure = |code: std::os::raw::c_int| {
            rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(code), None)
        };
        assert!(is_disk_full(&failure(rusqlite::ffi::SQLITE_FULL)));
        assert!(is_disk_full(&failure(rusqlite::ffi::SQLITE_IOERR)));
        assert!(!is_disk_full(&failure(rusqlite::ffi::SQLITE_BUSY)));
        assert!(!is_disk_full(&rusqlite::Error::QueryReturnedNoRows));
    }

    /// Low-disk degradation end-to-end (`repair_external_content_fts`,
    /// `LCM:db_bootstrap.py:533-545`): a rebuild that hits `SQLITE_FULL` (forced here by clamping
    /// `max_page_count`) drops the index + triggers and flags the store degraded — search routes
    /// to LIKE-only, base-table writes keep working — and a later repair with space available
    /// rebuilds the index and re-enables FTS.
    #[test]
    fn full_disk_rebuild_degrades_to_like_only_and_recovers() {
        let dir = std::env::temp_dir().join(format!("lcm-fts-degrade-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("lcm.db");
        let store = Store::open(&path).expect("open");
        assert!(!store.is_degraded());
        store
            .append_batch(
                "s1",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("indexed needle before the squeeze".into()),
                    token_estimate: 5,
                    ..Default::default()
                }],
                10.0,
            )
            .unwrap();
        {
            let conn = store.conn.lock().unwrap();
            // Desync the index (drop the sync trigger, write rows past it) with enough content
            // that reindexing must allocate pages beyond the clamped budget.
            conn.execute_batch("DROP TRIGGER msg_fts_insert;").unwrap();
            let filler = format!("needle {}", "expandable content ".repeat(60));
            {
                let mut stmt = conn
                    .prepare(
                        "INSERT INTO messages (session_id, source, role, content, timestamp) \
                         VALUES ('s1', 'cli', 'user', ?1, 11.0)",
                    )
                    .unwrap();
                for _ in 0..300 {
                    stmt.execute([&filler]).unwrap();
                }
            }
            let pages: i64 = conn
                .query_row("PRAGMA page_count", [], |r| r.get(0))
                .unwrap();
            conn.execute_batch(&format!("PRAGMA max_page_count = {pages}"))
                .unwrap();
        }

        // The forced repair detects the desync, attempts the rebuild, hits SQLITE_FULL, and
        // degrades instead of erroring.
        let outcomes = store.repair_fts(true).expect("degrades, does not error");
        assert!(
            outcomes.iter().any(|o| o.degraded && !o.rebuilt),
            "a degraded outcome was reported: {outcomes:?}"
        );
        assert!(store.is_degraded());

        // Restore disk headroom: writes keep working (the sync triggers are gone, so no insert
        // touches the missing index), and search routes to the LIKE fallback.
        store
            .conn
            .lock()
            .unwrap()
            .execute_batch("PRAGMA max_page_count = 1073741823")
            .unwrap();
        store
            .append_batch(
                "s1",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("written while degraded".into()),
                    token_estimate: 3,
                    ..Default::default()
                }],
                12.0,
            )
            .expect("writes work while degraded");
        let filter = MessageFilter {
            session: Some("s1"),
            role: None,
            source: None,
            time_from: None,
            time_to: None,
        };
        let like_hits =
            crate::search::search_messages(&store, "needle", SortMode::Recency, &filter, 10)
                .expect("degraded search stays available");
        assert!(!like_hits.is_empty(), "LIKE-only search finds content");

        // Recovery: a later repair pass rebuilds cleanly, re-enables FTS, and covers the rows
        // written while degraded.
        let outcomes = store.repair_fts(true).unwrap();
        assert!(outcomes.iter().any(|o| o.rebuilt), "index rebuilt");
        assert!(!store.is_degraded(), "degradation cleared");
        assert_eq!(
            store
                .search_messages("s1", "written while degraded", 10)
                .unwrap()
                .len(),
            1,
            "FTS covers rows written during degradation"
        );
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The open-path deep check is throttled by the recorded marker: a recent marker is left
    /// alone, a stale one refreshes when the check re-runs (`_should_run_integrity_check`,
    /// `LCM:db_bootstrap.py:386-398`).
    #[test]
    fn fts_deep_check_throttles_by_marker_age() {
        let dir = std::env::temp_dir().join(format!("lcm-fts-throttle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("lcm.db");
        let marker = |store: &Store| -> f64 {
            store
                .conn
                .lock()
                .unwrap()
                .query_row(
                    "SELECT value FROM metadata \
                     WHERE key = 'fts_integrity_checked_at:messages_fts'",
                    [],
                    |r| r.get::<_, String>(0),
                )
                .unwrap()
                .parse()
                .unwrap()
        };
        let first = {
            let store = Store::open(&path).expect("open");
            marker(&store)
        };
        let second = {
            let store = Store::open(&path).expect("reopen");
            marker(&store)
        };
        assert_eq!(first, second, "a fresh marker suppresses the deep check");

        {
            let store = Store::open(&path).expect("reopen");
            store
                .conn
                .lock()
                .unwrap()
                .execute(
                    "UPDATE metadata SET value = ?1 \
                     WHERE key = 'fts_integrity_checked_at:messages_fts'",
                    [(first - 25.0 * 3600.0).to_string()],
                )
                .unwrap();
        }
        let store = Store::open(&path).expect("reopen past interval");
        assert!(
            marker(&store) > first - 1.0,
            "a stale marker lets the deep check run and re-stamp"
        );
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
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

    /// `bind_session` parity (`LCM:lifecycle_state.py:123-227`): same-session rebind is a no-op;
    /// binding a different session zeroes the active frontier, preserves the finalized markers,
    /// and stamps `last_rollover_at`.
    #[test]
    fn bind_session_switch_resets_frontier_and_stamps_rollover() {
        let store = Store::open_in_memory().expect("open");
        let first = store.bind_session("conv", "s1", 1.0).unwrap();
        assert_eq!(first.current_session_id.as_deref(), Some("s1"));
        assert_eq!(first.current_bound_at, Some(1.0));
        store.advance_frontier("conv", 7, 2.0).unwrap();

        // Same-session rebind: no write (bound-at unchanged, frontier kept).
        let again = store.bind_session("conv", "s1", 3.0).unwrap();
        assert_eq!(again.current_bound_at, Some(1.0));
        assert_eq!(again.current_frontier_store_id, 7);

        store.finalize_session("conv", "s1", 7, 4.0).unwrap();
        let finalized = store.get_lifecycle("conv").unwrap().unwrap();
        assert_eq!(finalized.current_session_id, None);
        assert_eq!(finalized.last_finalized_session_id.as_deref(), Some("s1"));
        assert_eq!(finalized.last_finalized_frontier_store_id, 7);
        assert_eq!(finalized.last_finalized_at, Some(4.0));

        // A new session takes over the conversation: frontier zeroed, finalized markers kept,
        // rollover stamped (the finalized session differs from the incoming one).
        let switched = store.bind_session("conv", "s2", 5.0).unwrap();
        assert_eq!(switched.current_session_id.as_deref(), Some("s2"));
        assert_eq!(switched.current_frontier_store_id, 0);
        assert_eq!(switched.last_finalized_session_id.as_deref(), Some("s1"));
        assert_eq!(switched.last_finalized_frontier_store_id, 7);
        assert_eq!(switched.last_rollover_at, Some(5.0));
    }

    /// `finalize_session` on a non-current session keeps the active binding intact.
    #[test]
    fn finalize_other_session_keeps_current_binding() {
        let store = Store::open_in_memory().expect("open");
        store.bind_session("conv", "s2", 1.0).unwrap();
        store.advance_frontier("conv", 9, 2.0).unwrap();
        store.finalize_session("conv", "s1", 4, 3.0).unwrap();
        let row = store.get_lifecycle("conv").unwrap().unwrap();
        assert_eq!(row.current_session_id.as_deref(), Some("s2"));
        assert_eq!(row.current_frontier_store_id, 9);
        assert_eq!(row.last_finalized_session_id.as_deref(), Some("s1"));
        assert_eq!(row.last_finalized_frontier_store_id, 4);
    }

    /// `record_rollover` (`LCM:lifecycle_state.py:274-335`): one-shot old->new switch, idempotent.
    #[test]
    fn record_rollover_binds_new_and_finalizes_old() {
        let store = Store::open_in_memory().expect("open");
        store.record_rollover("conv", "s1", "s2", 12, 1.0).unwrap();
        let row = store.get_lifecycle("conv").unwrap().unwrap();
        assert_eq!(row.current_session_id.as_deref(), Some("s2"));
        assert_eq!(row.last_finalized_session_id.as_deref(), Some("s1"));
        assert_eq!(row.current_frontier_store_id, 0);
        assert_eq!(row.last_finalized_frontier_store_id, 12);
        assert_eq!(row.last_rollover_at, Some(1.0));
        // Idempotent: replaying the same rollover later does not restamp.
        store.record_rollover("conv", "s1", "s2", 12, 9.0).unwrap();
        let row2 = store.get_lifecycle("conv").unwrap().unwrap();
        assert_eq!(row2.last_rollover_at, Some(1.0));
    }

    /// `record_reset` stamps `last_reset_at` and clears deferred debt.
    #[test]
    fn record_reset_clears_debt() {
        let store = Store::open_in_memory().expect("open");
        store.bind_session("conv", "s1", 1.0).unwrap();
        store.record_debt("conv", "compaction", 500, 2.0).unwrap();
        store.record_maintenance_attempt("conv", 3.0).unwrap();
        store.record_reset("conv", 4.0).unwrap();
        let row = store.get_lifecycle("conv").unwrap().unwrap();
        assert_eq!(row.debt_kind, None);
        assert_eq!(row.debt_size_estimate, 0);
        assert_eq!(row.last_reset_at, Some(4.0));
        assert_eq!(row.last_maintenance_attempt_at, Some(3.0));
    }

    /// `prune_empty_sessions` (`LCM:lifecycle_state.py:638-753`): only rows whose referenced
    /// sessions have zero messages AND zero nodes are deleted; protected sessions and young rows
    /// survive.
    #[test]
    fn prune_empty_sessions_respects_data_protection_and_age() {
        let store = Store::open_in_memory().expect("open");
        // s-data has messages; s-empty, s-protected, s-young have nothing.
        store.bind_session("conv-data", "s-data", 100.0).unwrap();
        store.bind_session("conv-empty", "s-empty", 100.0).unwrap();
        store
            .bind_session("conv-protected", "s-protected", 100.0)
            .unwrap();
        store.bind_session("conv-young", "s-young", 990.0).unwrap();
        store
            .append_batch(
                "s-data",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("hello".into()),
                    ..NewMessage::default()
                }],
                101.0,
            )
            .unwrap();

        // Age guard: at now=1000 with a 0.1h (=360s) guard, rows bound at 100 are old enough;
        // conv-young (bound at 990) is too fresh.
        let deleted = store
            .prune_empty_sessions(&["s-protected".to_string()], Some(0.1), 1000.0)
            .unwrap();
        assert_eq!(deleted, 1, "only conv-empty pruned");
        assert!(store.get_lifecycle("conv-empty").unwrap().is_none());
        assert!(store.get_lifecycle("conv-data").unwrap().is_some());
        assert!(store.get_lifecycle("conv-protected").unwrap().is_some());
        assert!(store.get_lifecycle("conv-young").unwrap().is_some());
        assert_eq!(store.lifecycle_row_count().unwrap(), 3);

        // Without the age guard the young empty row goes too.
        let deleted = store
            .prune_empty_sessions(&["s-protected".to_string()], None, 1000.0)
            .unwrap();
        assert_eq!(deleted, 1);
        assert!(store.get_lifecycle("conv-young").unwrap().is_none());
    }

    /// A row whose *finalized* session still has data survives even when the current session is
    /// empty (either reference having data keeps the row).
    #[test]
    fn prune_keeps_rows_whose_finalized_session_has_data() {
        let store = Store::open_in_memory().expect("open");
        store.bind_session("conv", "s-old", 1.0).unwrap();
        store
            .append_batch(
                "s-old",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("kept".into()),
                    ..NewMessage::default()
                }],
                2.0,
            )
            .unwrap();
        store.finalize_session("conv", "s-old", 1, 3.0).unwrap();
        store.bind_session("conv", "s-new", 4.0).unwrap();
        let deleted = store.prune_empty_sessions(&[], None, 1000.0).unwrap();
        assert_eq!(deleted, 0, "finalized session still has messages");
        assert!(store.get_lifecycle("conv").unwrap().is_some());
    }

    /// `delete_below_depth` / `delete_session_nodes` / `reassign_session_{nodes,messages}`
    /// (`LCM:dag.py:233-270`, `LCM:store.py:358-368`).
    #[test]
    fn retain_depth_prune_and_session_reassignment() {
        let store = Store::open_in_memory().expect("open");
        for depth in [0, 0, 1, 2] {
            store
                .add_node(&NewNode {
                    session_id: "old".into(),
                    depth,
                    summary: format!("d{depth}"),
                    token_count: 1,
                    source_token_count: 2,
                    source_ids: vec![1],
                    source_type: if depth == 0 {
                        SourceType::Messages
                    } else {
                        SourceType::Nodes
                    },
                    created_at: 1.0,
                    earliest_at: None,
                    latest_at: None,
                    expand_hint: String::new(),
                })
                .unwrap();
        }
        store
            .append_batch(
                "old",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("raw".into()),
                    ..NewMessage::default()
                }],
                2.0,
            )
            .unwrap();

        // retain depth 2: both D0 leaves and the D1 node go; the D2 node stays.
        assert_eq!(store.delete_below_depth("old", 2).unwrap(), 3);
        assert_eq!(store.summary_count("old").unwrap(), 1);

        // Carry-over: the surviving node + raw messages move to the new session.
        assert_eq!(store.reassign_session_nodes("old", "new").unwrap(), 1);
        assert_eq!(store.reassign_session_messages("old", "new").unwrap(), 1);
        assert_eq!(store.summary_count("old").unwrap(), 0);
        assert_eq!(store.summary_count("new").unwrap(), 1);
        assert_eq!(store.message_count("old").unwrap(), 0);
        assert_eq!(store.message_count("new").unwrap(), 1);
        // Self/blank reassignment no-ops.
        assert_eq!(store.reassign_session_messages("new", "new").unwrap(), 0);

        // retain depth 0: everything goes.
        assert_eq!(store.delete_session_nodes("new").unwrap(), 1);
        assert_eq!(store.summary_count("new").unwrap(), 0);
    }

    /// `tail_min_store_id` (`get_session_tail` boundary probe, `LCM:engine.py:4542`): smallest
    /// store_id inside the newest-N window, `None` for an unknown session.
    #[test]
    fn tail_min_store_id_probes_the_rotate_boundary() {
        let store = Store::open_in_memory().expect("open");
        let msgs: Vec<NewMessage> = (0..6)
            .map(|i| NewMessage {
                role: "user".into(),
                content: Some(format!("m{i}")),
                ..NewMessage::default()
            })
            .collect();
        store.append_batch("s1", &msgs, 1.0).unwrap();
        assert_eq!(store.tail_min_store_id("s1", 2).unwrap(), Some(5));
        assert_eq!(
            store.tail_min_store_id("s1", 100).unwrap(),
            Some(1),
            "tail wider than the session covers everything"
        );
        assert_eq!(store.tail_min_store_id("missing", 2).unwrap(), None);
    }

    /// `backup_to` (`rusqlite::backup` online copy): the snapshot is an independently readable
    /// database with the same rows.
    #[test]
    fn backup_to_writes_a_readable_snapshot() {
        let dir = std::env::temp_dir().join(format!("lcm-backup-to-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open(dir.join("live.db")).expect("open");
        store
            .append_batch(
                "s1",
                &[
                    NewMessage {
                        role: "user".into(),
                        content: Some("first".into()),
                        ..NewMessage::default()
                    },
                    NewMessage {
                        role: "assistant".into(),
                        content: Some("second".into()),
                        ..NewMessage::default()
                    },
                ],
                1.0,
            )
            .unwrap();
        let dest = dir.join("snap.sqlite3");
        store.backup_to(&dest).unwrap();
        let copy = Store::open(&dest).expect("snapshot opens");
        assert_eq!(copy.message_count("s1").unwrap(), 2);
        drop(copy);
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `scan_fts_repair` (`_scan_fts_repair`, `LCM:command.py:665-701`) reports a desynced index
    /// without touching it; a forced `repair_fts` then heals it.
    #[test]
    fn scan_fts_repair_flags_desync_and_forced_repair_heals() {
        let store = Store::open_in_memory().expect("open");
        store
            .append_batch(
                "s1",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("indexed row".into()),
                    ..NewMessage::default()
                }],
                1.0,
            )
            .unwrap();
        {
            // Sabotage: drop the insert trigger, then write a row the index never sees.
            let conn = store.conn.lock().unwrap();
            conn.execute_batch("DROP TRIGGER msg_fts_insert;").unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, source, role, content, timestamp) \
                 VALUES ('s1', 'cli', 'user', 'unindexed row', 2.0)",
                [],
            )
            .unwrap();
        }

        let scans = store.scan_fts_repair();
        assert_eq!(scans.len(), 2);
        let messages = scans.iter().find(|s| s.table == "messages_fts").unwrap();
        assert!(messages.needs_repair, "desync detected");
        assert_eq!(messages.content_rows, Some(2));
        // COUNT(*) on an external-content FTS5 table reads through to the content table (same
        // in Python), so the row counts agree even while desynced — the desync verdict comes
        // from the `_docsize` shadow mismatch underneath `needs_repair`.
        assert_eq!(messages.fts_rows, Some(2));
        let nodes = scans.iter().find(|s| s.table == "nodes_fts").unwrap();
        assert!(!nodes.needs_repair, "untouched index stays clean");

        // The scan is read-only: the desync is still there until a forced repair.
        assert!(store.scan_fts_repair()[0].needs_repair);
        let repairs = store.repair_fts(true).unwrap();
        assert!(repairs.iter().any(|r| r.rebuilt));
        let scans = store.scan_fts_repair();
        assert!(scans.iter().all(|s| !s.needs_repair), "healed");
        assert_eq!(scans[0].content_rows, scans[0].fts_rows);
    }
}
