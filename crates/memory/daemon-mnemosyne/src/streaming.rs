// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Streaming memory + delta sync — port of `mnemosyne/core/streaming.py`.
//!
//! Two independent capabilities:
//! - [`MemoryStream`]: in-process pub/sub of [`MemoryEvent`]s (push callbacks + pull channels +
//!   a bounded replay buffer for late joiners). [`crate::Engine::enable_streaming`] wires it into
//!   the write path at the same three sites as `beam.py` `_emit_event`: dedup-update
//!   (`MEMORY_UPDATED`), new insert (`MEMORY_ADDED`), and consolidation output
//!   (`MEMORY_CONSOLIDATED`). Streaming failures never block memory operations.
//! - [`DeltaSync`]: row-level incremental mirror of `working_memory` / `episodic_memory`
//!   between banks, with per-`(peer, table)` JSON checkpoints. The C25 hardening carries over
//!   verbatim: a table allowlist resolved to `"main"."…"`-qualified SQL (temp-table shadowing),
//!   and *opt-in* column allowlists for peer mutations — identity, scope, lifecycle, and
//!   authorship columns are destination-controlled; sync is a content mirror, routing stays
//!   local.
//!
//! Rust-shape divergences (behavior preserved):
//! - Not feature-gated (the spec sketched this under `sync`): `MemoryStream`/`DeltaSync` are
//!   pure std + the crate's mandatory deps, so a gate bought nothing — same call as `dr.rs`.
//!   The heavy replication half (`sync.rs`: crypto + HTTP) stays behind `sync`.
//! - Callback deregistration is by [`SubscriptionId`], not callback identity (Rust closures
//!   are not comparable); `listen` returns an `mpsc::Receiver` instead of Python's busy-wait
//!   iterator. Callbacks still run *outside* the stream lock, so a subscriber may re-enter
//!   (emit/subscribe) without deadlocking, and a panicking callback is contained.
//! - BLOB delta values (`binary_vector`) transport as base64 text and decode back to bytes on
//!   apply, so a Rust↔Rust delta round-trips binaries even over JSON transport (Python only
//!   round-trips bytes in-process; over its JSON server the bytes were mangled by `default=str`).
//! - Checkpoints default to the bank-adjacent `<data_dir>/sync/` (Python:
//!   `~/.hermes/mnemosyne/sync/`) — the node never resolves `$HOME`. The filename scheme is
//!   Python's, including the legacy no-table-suffix fallback read.

use crate::engine::Engine;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Tables DeltaSync may operate on (C25 allowlist, `ALLOWED_DELTA_TABLES`).
pub const ALLOWED_DELTA_TABLES: &[&str] = &["working_memory", "episodic_memory"];

/// Resolve an allowlisted table to its `main.`-qualified SQL form (`_validate_table`; C25
/// hardening — SQLite resolves unqualified names to the temp schema first, so a same-connection
/// `CREATE TEMP TABLE working_memory` shadow would otherwise capture the sync SQL). Everything
/// downstream must use the returned string, never the raw input.
fn validate_table(table: &str, method: &str) -> Result<&'static str> {
    match table {
        "working_memory" => Ok("\"main\".\"working_memory\""),
        "episodic_memory" => Ok("\"main\".\"episodic_memory\""),
        other => Err(Error::Invalid(format!(
            "DeltaSync.{method}: table {other:?} is not in the allowlist \
             {ALLOWED_DELTA_TABLES:?}; adding a syncable table is a deliberate change to \
             streaming.rs, not a ride-along via the argument"
        ))),
    }
}

/// Columns a peer may mutate on an existing row (`_DELTA_UPDATABLE_COLUMNS`): content +
/// sync-relevant metadata only. Identity (`id`), scope (`session_id`, `scope`), lifecycle
/// (`valid_until`, `superseded_by`, timestamps, tier, recall counters), and authorship
/// (`author_id`, `author_type`, `channel_id`) are all destination-controlled.
const DELTA_UPDATABLE_COLUMNS: &[&str] = &[
    "content",
    "importance",
    "metadata_json",
    "veracity",
    "memory_type",
    "binary_vector",
    "source",
    "summary_of",
];

/// Columns a peer may supply when creating a row (`_DELTA_INSERTABLE_COLUMNS`): the updatable
/// set + row identity (`id`) + the peer's original creation `timestamp` (preserves history).
/// Destination defaults fill everything else, so a peer cannot land a row inside the local
/// session, claim authorship, or pre-tombstone a future legitimate write.
const DELTA_INSERTABLE_COLUMNS: &[&str] = &[
    "id",
    "content",
    "importance",
    "metadata_json",
    "veracity",
    "memory_type",
    "binary_vector",
    "source",
    "summary_of",
    "timestamp",
];

// ── MemoryStream ────────────────────────────────────────────────────────────────────────────

/// Memory-system event kinds (`EventType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// A new memory row was stored.
    #[serde(rename = "MEMORY_ADDED")]
    MemoryAdded,
    /// A memory was returned by recall (declared by Python; never emitted by either impl).
    #[serde(rename = "MEMORY_RECALLED")]
    MemoryRecalled,
    /// A memory was invalidated / soft-deleted (declared by Python; never emitted).
    #[serde(rename = "MEMORY_INVALIDATED")]
    MemoryInvalidated,
    /// Working rows were consolidated into an episodic summary.
    #[serde(rename = "MEMORY_CONSOLIDATED")]
    MemoryConsolidated,
    /// An existing memory row was refreshed in place (exact-content dedup).
    #[serde(rename = "MEMORY_UPDATED")]
    MemoryUpdated,
}

/// One memory-system event (`MemoryEvent`). Serializes with the Python field set and the
/// `EventType.name` string (`to_dict` L117-L120).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryEvent {
    /// What happened.
    pub event_type: EventType,
    /// The affected memory row id.
    pub memory_id: String,
    /// Emission time (ISO).
    pub timestamp: String,
    /// Session the mutation ran under.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Row content (when the event carries it).
    #[serde(default)]
    pub content: Option<String>,
    /// Ingestion source.
    #[serde(default)]
    pub source: Option<String>,
    /// Row importance.
    #[serde(default)]
    pub importance: Option<f64>,
    /// Caller metadata.
    #[serde(default)]
    pub metadata: Option<Value>,
    /// Only the changed fields, for update events.
    #[serde(default)]
    pub delta: Option<Value>,
}

/// Handle for deregistering a stream callback ([`MemoryStream::off`]; the Rust replacement for
/// Python's remove-by-callback-identity `off`/`off_any`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubscriptionId(u64);

type Callback = Arc<dyn Fn(&MemoryEvent) + Send + Sync>;

struct StreamInner {
    next_id: u64,
    /// `(id, filter, callback)`; `None` filter = any-event callback (`on_any`).
    callbacks: Vec<(u64, Option<EventType>, Callback)>,
    /// `(filter, sender)`; senders whose receiver hung up are pruned on emit.
    listeners: Vec<(Option<Vec<EventType>>, Sender<MemoryEvent>)>,
    buffer: VecDeque<MemoryEvent>,
    max_buffer: usize,
}

/// Real-time event stream for memory operations (`MemoryStream`): push (callbacks) and pull
/// (channels) with a bounded replay buffer for late joiners. Thread-safe.
pub struct MemoryStream {
    inner: Mutex<StreamInner>,
}

impl Default for MemoryStream {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl MemoryStream {
    /// A stream buffering at most `max_buffer` events (`__init__`, default 1000).
    pub fn new(max_buffer: usize) -> Self {
        Self {
            inner: Mutex::new(StreamInner {
                next_id: 0,
                callbacks: Vec::new(),
                listeners: Vec::new(),
                buffer: VecDeque::new(),
                max_buffer,
            }),
        }
    }

    /// Register a callback for one event type (`on`).
    pub fn on(
        &self,
        event_type: EventType,
        callback: impl Fn(&MemoryEvent) + Send + Sync + 'static,
    ) -> SubscriptionId {
        self.register(Some(event_type), Arc::new(callback))
    }

    /// Register a callback for all event types (`on_any`).
    pub fn on_any(
        &self,
        callback: impl Fn(&MemoryEvent) + Send + Sync + 'static,
    ) -> SubscriptionId {
        self.register(None, Arc::new(callback))
    }

    fn register(&self, filter: Option<EventType>, callback: Callback) -> SubscriptionId {
        let mut inner = self.inner.lock().unwrap();
        inner.next_id += 1;
        let id = inner.next_id;
        inner.callbacks.push((id, filter, callback));
        SubscriptionId(id)
    }

    /// Remove a callback by subscription handle (`off` / `off_any`).
    pub fn off(&self, id: SubscriptionId) {
        self.inner
            .lock()
            .unwrap()
            .callbacks
            .retain(|(cid, _, _)| *cid != id.0);
    }

    /// Emit an event to the replay buffer, pull listeners, and callbacks (`emit`). Like Python,
    /// callbacks run *outside* the lock (a subscriber may re-enter the stream) and a callback
    /// failure (panic) never breaks the stream or the memory operation behind it.
    pub fn emit(&self, event: MemoryEvent) {
        let matching: Vec<Callback> = {
            let mut inner = self.inner.lock().unwrap();
            inner.buffer.push_back(event.clone());
            while inner.buffer.len() > inner.max_buffer {
                inner.buffer.pop_front();
            }
            // Channel sends run no user code, so they stay inside the lock; hung-up listeners
            // prune here (Python's iterators deregister in __del__).
            inner.listeners.retain(|(filter, tx)| {
                let wanted = filter
                    .as_ref()
                    .is_none_or(|types| types.contains(&event.event_type));
                !wanted || tx.send(event.clone()).is_ok()
            });
            inner
                .callbacks
                .iter()
                .filter(|(_, filter, _)| filter.is_none_or(|t| t == event.event_type))
                .map(|(_, _, cb)| cb.clone())
                .collect()
        };
        for cb in matching {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cb(&event)));
        }
    }

    /// A pull-style subscription: events arrive on the returned channel as they occur
    /// (`listen`; the Python busy-wait iterator becomes a receiver — drop it to unsubscribe).
    pub fn listen(&self, event_types: Option<Vec<EventType>>) -> Receiver<MemoryEvent> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.inner.lock().unwrap().listeners.push((event_types, tx));
        rx
    }

    /// Buffered events, optionally filtered by type and `timestamp >= since` (`get_buffer`).
    pub fn get_buffer(
        &self,
        event_types: Option<&[EventType]>,
        since: Option<&str>,
    ) -> Vec<MemoryEvent> {
        self.inner
            .lock()
            .unwrap()
            .buffer
            .iter()
            .filter(|e| event_types.is_none_or(|types| types.contains(&e.event_type)))
            .filter(|e| since.is_none_or(|s| e.timestamp.as_str() >= s))
            .cloned()
            .collect()
    }

    /// Drop all buffered events (`clear_buffer`).
    pub fn clear_buffer(&self) {
        self.inner.lock().unwrap().buffer.clear();
    }
}

// ── DeltaSync ───────────────────────────────────────────────────────────────────────────────

/// Checkpoint for incremental delta sync (`SyncCheckpoint`), persisted as JSON per
/// `(peer, table)` — rowid namespaces are table-local, hence the per-table scoping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCheckpoint {
    /// The peer this checkpoint tracks.
    pub peer_id: String,
    /// When the last sync completed (ISO).
    pub last_sync_at: String,
    /// Last synced memory id (informational).
    #[serde(default)]
    pub last_memory_id: Option<String>,
    /// High-water rowid in the scoped table.
    #[serde(default)]
    pub last_rowid: i64,
}

/// Incremental row-level synchronization between two banks (`DeltaSync`): only rows changed
/// since the last checkpoint transfer, and only allowlisted columns apply.
pub struct DeltaSync<'e> {
    engine: &'e Engine,
    checkpoint_dir: PathBuf,
    checkpoints: Mutex<HashMap<(String, String), SyncCheckpoint>>,
    /// Per-table live-schema column sets, cached after the first `PRAGMA main.table_info`
    /// (`_column_cache`; a PRAGMA per row would dominate `apply_delta`).
    column_cache: Mutex<HashMap<String, HashSet<String>>>,
}

impl<'e> DeltaSync<'e> {
    /// A delta-syncer for `engine`, with checkpoints under `checkpoint_dir` (default
    /// `<data_dir>/sync`). Existing checkpoint files — including legacy per-peer-only names —
    /// load eagerly (`__init__` + `_load_checkpoints`).
    pub fn new(engine: &'e Engine, checkpoint_dir: Option<PathBuf>) -> Self {
        let dir = checkpoint_dir.unwrap_or_else(|| engine.config().data_dir.join("sync"));
        let ds = Self {
            engine,
            checkpoint_dir: dir,
            checkpoints: Mutex::new(HashMap::new()),
            column_cache: Mutex::new(HashMap::new()),
        };
        ds.load_checkpoints();
        ds
    }

    /// `checkpoint_<peer>__<table>.json` (`_checkpoint_path`; the double underscore avoids
    /// collisions with peer ids containing single underscores).
    fn checkpoint_path(&self, peer_id: &str, table: &str) -> PathBuf {
        self.checkpoint_dir
            .join(format!("checkpoint_{peer_id}__{table}.json"))
    }

    /// Parse `checkpoint_<peer>__<table>` or the legacy `checkpoint_<peer>` form
    /// (`_parse_checkpoint_filename`; legacy maps to `working_memory`).
    fn parse_checkpoint_stem(stem: &str) -> Option<(String, String)> {
        let body = stem.strip_prefix("checkpoint_")?;
        match body.rsplit_once("__") {
            Some((peer, table)) if !peer.is_empty() && !table.is_empty() => {
                Some((peer.to_string(), table.to_string()))
            }
            Some(_) => None,
            None => Some((body.to_string(), "working_memory".to_string())),
        }
    }

    fn load_checkpoints(&self) {
        let Ok(read) = std::fs::read_dir(&self.checkpoint_dir) else {
            return;
        };
        let mut cps = self.checkpoints.lock().unwrap();
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(key) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(Self::parse_checkpoint_stem)
            else {
                continue;
            };
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(cp) = serde_json::from_str::<SyncCheckpoint>(&raw) {
                    cps.insert(key, cp);
                }
            }
        }
    }

    /// The current checkpoint for a `(peer, table)` pair (`get_checkpoint`).
    pub fn get_checkpoint(&self, peer_id: &str, table: &str) -> Option<SyncCheckpoint> {
        self.checkpoints
            .lock()
            .unwrap()
            .get(&(peer_id.to_string(), table.to_string()))
            .cloned()
    }

    /// Set and persist a checkpoint (`set_checkpoint`). Persistence is best-effort — an
    /// unwritable checkpoint dir degrades to per-process checkpoints, never an apply failure.
    pub fn set_checkpoint(&self, peer_id: &str, checkpoint: SyncCheckpoint, table: &str) {
        self.checkpoints
            .lock()
            .unwrap()
            .insert((peer_id.to_string(), table.to_string()), checkpoint.clone());
        if std::fs::create_dir_all(&self.checkpoint_dir).is_ok() {
            let path = self.checkpoint_path(peer_id, table);
            if let Ok(body) = serde_json::to_string(&checkpoint) {
                if let Err(e) = std::fs::write(&path, body) {
                    tracing::debug!(error = %e, "sync checkpoint write failed (non-fatal)");
                }
            }
        }
    }

    /// Schema-derived column set for `table` (`_allowed_columns`). Defense-in-depth on top of
    /// the static opt-in sets: an applied column must be in BOTH the allowlist AND the live
    /// schema (`PRAGMA main.table_info`, qualified against temp shadowing).
    fn schema_columns(&self, table: &str) -> Result<HashSet<String>> {
        validate_table(table, "schema_columns")?;
        if let Some(cols) = self.column_cache.lock().unwrap().get(table) {
            return Ok(cols.clone());
        }
        let cols: HashSet<String> = self.engine.with_conn(|conn| {
            let mut stmt = conn.prepare(&format!("PRAGMA main.table_info(\"{table}\")"))?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
            rows.collect::<std::result::Result<HashSet<_>, _>>()
                .map_err(Error::from)
        })?;
        if cols.is_empty() {
            return Err(Error::Invalid(format!(
                "DeltaSync: PRAGMA main.table_info({table:?}) returned no columns — the table \
                 is allowlisted but the schema is missing"
            )));
        }
        self.column_cache
            .lock()
            .unwrap()
            .insert(table.to_string(), cols.clone());
        Ok(cols)
    }

    /// Rows changed since the last checkpoint with `peer_id` (`compute_delta`): full row objects
    /// where `rowid > cp.last_rowid OR timestamp > cp.last_sync_at`, everything on first sync.
    /// The internal `embedding` field is stripped (L475); BLOBs export as base64 text.
    pub fn compute_delta(&self, peer_id: &str, table: &str) -> Result<Vec<Value>> {
        let qualified = validate_table(table, "compute_delta")?;
        let checkpoint = self.get_checkpoint(peer_id, table);
        self.engine.with_conn(|conn| {
            let (sql, params): (String, Vec<String>) = match &checkpoint {
                Some(cp) => (
                    format!(
                        "SELECT * FROM {qualified} WHERE rowid > ?1 OR timestamp > ?2 \
                         ORDER BY rowid ASC"
                    ),
                    vec![cp.last_rowid.to_string(), cp.last_sync_at.clone()],
                ),
                None => (
                    format!("SELECT * FROM {qualified} ORDER BY rowid ASC"),
                    Vec::new(),
                ),
            };
            let mut stmt = conn.prepare(&sql)?;
            let names: Vec<String> = stmt
                .column_names()
                .into_iter()
                .map(str::to_string)
                .collect();
            let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
            let mut delta = Vec::new();
            while let Some(row) = rows.next()? {
                let mut obj = Map::new();
                for (i, name) in names.iter().enumerate() {
                    if name == "embedding" {
                        continue;
                    }
                    obj.insert(name.clone(), value_ref_to_json(row.get_ref(i)?));
                }
                delta.push(Value::Object(obj));
            }
            Ok(delta)
        })
    }

    /// Apply an incoming delta from a peer (`apply_delta`). Returns
    /// `{inserted, updated, skipped, filtered_keys}` — `filtered_keys` counts peer-supplied
    /// keys dropped by the column allowlists (pre-C25 those keys crashed the apply or landed
    /// in SQL); the rest of the row still applies.
    pub fn apply_delta(&self, peer_id: &str, delta: &[Value], table: &str) -> Result<Value> {
        let qualified = validate_table(table, "apply_delta")?;
        let schema_cols = self.schema_columns(table)?;
        let updatable: HashSet<&str> = DELTA_UPDATABLE_COLUMNS
            .iter()
            .copied()
            .filter(|c| schema_cols.contains(*c))
            .collect();
        let insertable: HashSet<&str> = DELTA_INSERTABLE_COLUMNS
            .iter()
            .copied()
            .filter(|c| schema_cols.contains(*c))
            .collect();

        let (mut inserted, mut updated, mut skipped, mut filtered_keys) = (0u64, 0u64, 0u64, 0u64);
        self.engine.with_conn(|conn| {
            for mem in delta {
                let Some(obj) = mem.as_object() else {
                    skipped += 1;
                    continue;
                };
                let Some(mid) = obj.get("id").and_then(|v| v.as_str()) else {
                    skipped += 1;
                    continue;
                };
                let exists = conn
                    .query_row(
                        &format!("SELECT 1 FROM {qualified} WHERE id = ?1"),
                        [mid],
                        |_| Ok(()),
                    )
                    .is_ok();
                if exists {
                    // UPDATE: opt-in mutable columns only; None values skip (Python L537).
                    let mut cols: Vec<(&str, rusqlite::types::Value)> = Vec::new();
                    for (k, v) in obj {
                        if k == "id" {
                            continue; // match key, not a mutation target
                        }
                        if !updatable.contains(k.as_str()) {
                            filtered_keys += 1;
                            continue;
                        }
                        if v.is_null() {
                            continue;
                        }
                        cols.push((k, json_to_sql_value(k, v)));
                    }
                    if cols.is_empty() {
                        skipped += 1;
                        continue;
                    }
                    let sets = cols
                        .iter()
                        .enumerate()
                        .map(|(i, (k, _))| format!("\"{k}\" = ?{}", i + 1))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let mut params: Vec<rusqlite::types::Value> =
                        cols.into_iter().map(|(_, v)| v).collect();
                    let where_slot = params.len() + 1;
                    params.push(rusqlite::types::Value::Text(mid.to_string()));
                    conn.execute(
                        &format!("UPDATE {qualified} SET {sets} WHERE id = ?{where_slot}"),
                        rusqlite::params_from_iter(params),
                    )?;
                    updated += 1;
                } else {
                    // INSERT: opt-in columns; id must be present (peer supplies row identity).
                    let mut cols: Vec<(&str, rusqlite::types::Value)> = Vec::new();
                    for (k, v) in obj {
                        if !insertable.contains(k.as_str()) {
                            filtered_keys += 1;
                            continue;
                        }
                        cols.push((k, json_to_sql_value(k, v)));
                    }
                    if cols.is_empty() {
                        skipped += 1;
                        continue;
                    }
                    let quoted = cols
                        .iter()
                        .map(|(k, _)| format!("\"{k}\""))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let placeholders = (1..=cols.len())
                        .map(|i| format!("?{i}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let params: Vec<rusqlite::types::Value> =
                        cols.into_iter().map(|(_, v)| v).collect();
                    conn.execute(
                        &format!("INSERT INTO {qualified} ({quoted}) VALUES ({placeholders})"),
                        rusqlite::params_from_iter(params),
                    )?;
                    inserted += 1;
                }
            }
            Ok(())
        })?;

        // Advance the (peer, table) checkpoint to the table's high-water rowid.
        let max_rowid: i64 = self
            .engine
            .with_conn(|conn| {
                Ok(
                    conn.query_row(&format!("SELECT MAX(rowid) FROM {qualified}"), [], |r| {
                        r.get::<_, Option<i64>>(0)
                    })?,
                )
            })?
            .unwrap_or(0);
        self.set_checkpoint(
            peer_id,
            SyncCheckpoint {
                peer_id: peer_id.to_string(),
                last_sync_at: crate::util::now_iso(),
                last_memory_id: None,
                last_rowid: max_rowid,
            },
            table,
        );

        Ok(json!({
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "filtered_keys": filtered_keys,
        }))
    }

    /// Outbound half of a sync cycle (`sync_to`): the delta for `peer_id` plus the current
    /// checkpoint; the caller transports it.
    pub fn sync_to(&self, peer_id: &str, table: &str) -> Result<Value> {
        let delta = self.compute_delta(peer_id, table)?;
        let cp = self.get_checkpoint(peer_id, table);
        Ok(json!({
            "peer_id": peer_id,
            "table": table,
            "count": delta.len(),
            "delta": delta,
            "checkpoint": cp,
        }))
    }

    /// Inbound half of a sync cycle (`sync_from`): apply a peer's delta, report stats +
    /// checkpoint.
    pub fn sync_from(&self, peer_id: &str, delta: &[Value], table: &str) -> Result<Value> {
        let stats = self.apply_delta(peer_id, delta, table)?;
        let cp = self.get_checkpoint(peer_id, table);
        Ok(json!({
            "peer_id": peer_id,
            "table": table,
            "stats": stats,
            "checkpoint": cp,
        }))
    }
}

/// SQLite value -> JSON for delta rows. BLOBs become base64 text (Python keeps raw bytes
/// in-process; over JSON transport Rust's base64 round-trips where Python's `default=str`
/// mangled them).
fn value_ref_to_json(v: rusqlite::types::ValueRef<'_>) -> Value {
    use base64::Engine as _;
    use rusqlite::types::ValueRef;
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => json!(i),
        ValueRef::Real(f) => json!(f),
        ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
        ValueRef::Blob(b) => json!(base64::engine::general_purpose::STANDARD.encode(b)),
    }
}

/// JSON -> SQLite value for applying deltas. `binary_vector` (the one BLOB column in the
/// allowlists) decodes base64 back to bytes; everything else binds naturally.
fn json_to_sql_value(column: &str, v: &Value) -> rusqlite::types::Value {
    use base64::Engine as _;
    use rusqlite::types::Value as Sql;
    match v {
        Value::Null => Sql::Null,
        Value::Bool(b) => Sql::Integer(i64::from(*b)),
        Value::Number(n) => n
            .as_i64()
            .map(Sql::Integer)
            .unwrap_or_else(|| Sql::Real(n.as_f64().unwrap_or(0.0))),
        Value::String(s) => {
            if column == "binary_vector" {
                if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s) {
                    return Sql::Blob(bytes);
                }
            }
            Sql::Text(s.clone())
        }
        other => Sql::Text(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MnemosyneConfig;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn event(t: EventType, id: &str) -> MemoryEvent {
        MemoryEvent {
            event_type: t,
            memory_id: id.to_string(),
            timestamp: crate::util::now_iso(),
            session_id: None,
            content: None,
            source: None,
            importance: None,
            metadata: None,
            delta: None,
        }
    }

    #[test]
    fn stream_callbacks_filter_by_type_and_deregister() {
        let stream = MemoryStream::default();
        let added = Arc::new(AtomicUsize::new(0));
        let any = Arc::new(AtomicUsize::new(0));
        let a = added.clone();
        let sub = stream.on(EventType::MemoryAdded, move |_| {
            a.fetch_add(1, Ordering::SeqCst);
        });
        let b = any.clone();
        stream.on_any(move |_| {
            b.fetch_add(1, Ordering::SeqCst);
        });

        stream.emit(event(EventType::MemoryAdded, "m1"));
        stream.emit(event(EventType::MemoryUpdated, "m1"));
        assert_eq!(added.load(Ordering::SeqCst), 1, "typed callback filters");
        assert_eq!(any.load(Ordering::SeqCst), 2, "any-callback sees all");

        stream.off(sub);
        stream.emit(event(EventType::MemoryAdded, "m2"));
        assert_eq!(added.load(Ordering::SeqCst), 1, "off() deregisters");
        assert_eq!(any.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn stream_listen_and_buffer() {
        let stream = MemoryStream::new(3);
        let rx = stream.listen(Some(vec![EventType::MemoryConsolidated]));
        for i in 0..5 {
            stream.emit(event(EventType::MemoryAdded, &format!("m{i}")));
        }
        stream.emit(event(EventType::MemoryConsolidated, "summary"));

        // The pull channel got only the filtered event.
        let got = rx.try_recv().expect("one event");
        assert_eq!(got.memory_id, "summary");
        assert!(rx.try_recv().is_err());

        // The buffer is bounded at 3 and filterable.
        let all = stream.get_buffer(None, None);
        assert_eq!(all.len(), 3, "ring buffer bounded");
        let consolidated = stream.get_buffer(Some(&[EventType::MemoryConsolidated]), None);
        assert_eq!(consolidated.len(), 1);
        stream.clear_buffer();
        assert!(stream.get_buffer(None, None).is_empty());
    }

    #[test]
    fn panicking_and_reentrant_callbacks_never_break_the_stream() {
        let stream = Arc::new(MemoryStream::default());
        let after = Arc::new(AtomicUsize::new(0));
        stream.on(EventType::MemoryAdded, |_| panic!("hostile observer"));
        // Re-entrant subscriber: registering from inside a callback must not deadlock.
        let inner_stream = stream.clone();
        stream.on(EventType::MemoryAdded, move |_| {
            inner_stream.on_any(|_| {});
        });
        let a = after.clone();
        stream.on_any(move |_| {
            a.fetch_add(1, Ordering::SeqCst);
        });
        stream.emit(event(EventType::MemoryAdded, "m1"));
        assert_eq!(after.load(Ordering::SeqCst), 1, "later callbacks still ran");
    }

    #[test]
    fn event_serialization_uses_python_names() {
        let e = event(EventType::MemoryConsolidated, "m1");
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["event_type"], "MEMORY_CONSOLIDATED");
        let back: MemoryEvent = serde_json::from_value(v).unwrap();
        assert_eq!(back.event_type, EventType::MemoryConsolidated);
    }

    #[test]
    fn engine_emits_added_updated_consolidated() {
        let engine = Engine::open_in_memory(MnemosyneConfig::default()).expect("engine");
        let stream = engine.enable_streaming();
        let id = engine
            .remember("Maya moved to Lisbon", &Default::default())
            .expect("remember");
        // An exact-content re-remember takes the dedup-update path.
        engine
            .remember("Maya moved to Lisbon", &Default::default())
            .expect("dedup update");

        let buffer = stream.get_buffer(None, None);
        let kinds: Vec<EventType> = buffer.iter().map(|e| e.event_type).collect();
        assert_eq!(
            kinds,
            vec![EventType::MemoryAdded, EventType::MemoryUpdated],
            "one add + one dedup update: {buffer:?}"
        );
        assert_eq!(buffer[0].memory_id, id);
        assert_eq!(buffer[1].memory_id, id);
        assert_eq!(buffer[0].content.as_deref(), Some("Maya moved to Lisbon"));
        assert_eq!(buffer[0].session_id.as_deref(), Some("default"));

        let report = engine.sleep(true).expect("sleep");
        assert!(report.items_consolidated > 0, "{report:?}");
        let kinds: Vec<EventType> = stream
            .get_buffer(None, None)
            .iter()
            .map(|e| e.event_type)
            .collect();
        assert!(
            kinds.contains(&EventType::MemoryConsolidated),
            "sleep emits MEMORY_CONSOLIDATED: {kinds:?}"
        );
    }

    fn bank(dir: &Path) -> Engine {
        Engine::open(MnemosyneConfig {
            data_dir: dir.to_path_buf(),
            ..Default::default()
        })
        .expect("engine")
    }

    #[test]
    fn delta_round_trip_between_two_banks() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let a = bank(&tmp.path().join("a"));
        let b = bank(&tmp.path().join("b"));
        a.remember("row one from a", &Default::default()).unwrap();
        a.remember("row two from a", &Default::default()).unwrap();

        let ds_a = DeltaSync::new(&a, None);
        let ds_b = DeltaSync::new(&b, None);

        let out = ds_a.sync_to("bank-b", "working_memory").expect("sync_to");
        assert_eq!(out["count"], 2);
        let delta: Vec<Value> = out["delta"].as_array().unwrap().clone();
        let result = ds_b
            .sync_from("bank-a", &delta, "working_memory")
            .expect("sync_from");
        assert_eq!(result["stats"]["inserted"], 2);
        // Destination-controlled keys (session/scope/lifecycle/authorship) were filtered.
        assert!(result["stats"]["filtered_keys"].as_u64().unwrap() > 0);

        // Incremental: a checkpoint past the high-water mark yields an empty delta.
        ds_a.set_checkpoint(
            "bank-b",
            SyncCheckpoint {
                peer_id: "bank-b".into(),
                last_sync_at: crate::util::now_iso(),
                last_memory_id: None,
                last_rowid: i64::MAX,
            },
            "working_memory",
        );
        assert!(ds_a
            .compute_delta("bank-b", "working_memory")
            .expect("delta")
            .is_empty());

        // Re-applying the same delta updates instead of duplicating.
        let stats2 = ds_b
            .apply_delta("bank-a", &delta, "working_memory")
            .expect("re-apply");
        assert_eq!(stats2["inserted"], 0);
        assert_eq!(stats2["updated"], 2);

        // The mirrored rows are real rows: FTS finds them in bank b.
        let hits = b.recall("row one", 5).expect("recall");
        assert!(!hits.is_empty(), "delta-applied row is searchable");
    }

    #[test]
    fn delta_rejects_unlisted_tables_and_filters_hostile_columns() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let a = bank(tmp.path());
        let ds = DeltaSync::new(&a, None);
        assert!(ds.compute_delta("p", "sqlite_master").is_err());
        assert!(ds.compute_delta("p", "audit_log").is_err());
        assert!(ds.apply_delta("p", &[], "memory_events").is_err());

        // A hostile delta rerouting scope/session/authorship: those keys drop, the row still
        // lands with destination defaults.
        let hostile = json!({
            "id": "evil1",
            "content": "legit content",
            "session_id": "victim-session",
            "superseded_by": "someone-elses-row",
            "author_id": "forged",
        });
        let stats = ds
            .apply_delta("peer", &[hostile], "working_memory")
            .expect("apply");
        assert_eq!(stats["inserted"], 1);
        assert_eq!(stats["filtered_keys"], 3);
        let (session, superseded) = a
            .with_conn(|conn| {
                Ok(conn.query_row(
                    "SELECT session_id, superseded_by FROM working_memory WHERE id = 'evil1'",
                    [],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
                )?)
            })
            .expect("row");
        assert_eq!(session, "default", "destination controls session routing");
        assert_eq!(superseded, None, "peer cannot pre-tombstone");
    }

    #[test]
    fn delta_binary_vector_round_trips_as_base64() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let a = bank(&tmp.path().join("a"));
        let b = bank(&tmp.path().join("b"));
        let blob: Vec<u8> = vec![0xde, 0xad, 0x00, 0xff, 0x10];
        a.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodic_memory (id, content, timestamp, session_id, binary_vector) \
                 VALUES ('ep1', 'summary', ?1, 'default', ?2)",
                rusqlite::params![crate::util::now_iso(), blob],
            )?;
            Ok(())
        })
        .unwrap();

        let ds_a = DeltaSync::new(&a, None);
        let ds_b = DeltaSync::new(&b, None);
        let delta = ds_a.compute_delta("b", "episodic_memory").expect("delta");
        assert_eq!(delta.len(), 1);
        assert!(
            delta[0]["binary_vector"].is_string(),
            "blob exported as b64"
        );
        ds_b.apply_delta("a", &delta, "episodic_memory")
            .expect("apply");
        let restored: Vec<u8> = b
            .with_conn(|conn| {
                Ok(conn.query_row(
                    "SELECT binary_vector FROM episodic_memory WHERE id = 'ep1'",
                    [],
                    |r| r.get(0),
                )?)
            })
            .expect("blob");
        assert_eq!(restored, blob, "bytes identical after JSON transport");
    }

    #[test]
    fn checkpoints_persist_and_legacy_names_load() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let a = bank(tmp.path());
        {
            let ds = DeltaSync::new(&a, None);
            ds.set_checkpoint(
                "peer_x",
                SyncCheckpoint {
                    peer_id: "peer_x".into(),
                    last_sync_at: "2026-01-01T00:00:00".into(),
                    last_memory_id: None,
                    last_rowid: 42,
                },
                "episodic_memory",
            );
        }
        // A legacy file without a table suffix maps to working_memory.
        let legacy = tmp.path().join("sync").join("checkpoint_oldpeer.json");
        std::fs::write(
            &legacy,
            r#"{"peer_id":"oldpeer","last_sync_at":"2025-01-01T00:00:00","last_rowid":7}"#,
        )
        .unwrap();

        let ds = DeltaSync::new(&a, None);
        assert_eq!(
            ds.get_checkpoint("peer_x", "episodic_memory")
                .unwrap()
                .last_rowid,
            42
        );
        assert_eq!(
            ds.get_checkpoint("oldpeer", "working_memory")
                .unwrap()
                .last_rowid,
            7
        );
        assert!(ds.get_checkpoint("peer_x", "working_memory").is_none());
    }
}
