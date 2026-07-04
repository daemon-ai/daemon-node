// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Event-log memory replication — port of `mnemosyne/core/sync.py` (`sync` feature).
//!
//! [`SyncEngine`] orchestrates synchronization between banks on top of the append-only
//! `memory_events` table the write path already populates: [`SyncEngine::log_event`] records
//! mutations, [`SyncEngine::pull_changes`] pages them out by timestamp cursor,
//! [`SyncEngine::push_changes`] validates/dedups/conflict-resolves incoming events and applies
//! them through the **full** write pipeline ([`Engine::remember`] — FTS, dedup, knowledge
//! ingestion — exactly Python's `remember(memory_id=...)` routing), and
//! [`SyncEngine::sync_with`] runs a push/pull cycle against a remote sync server (a Python
//! `sync_server.py` peer, or any host mounting [`endpoints`]).
//! [`ConflictResolution`] implements v1 LWW (`timestamp` -> `importance` -> `device_id`) and the
//! v2 causal-chain variant over `parent_event_ids`. [`SyncEncryption`] seals payloads.
//!
//! Rust-shape divergences (documented in the port spec §12.1):
//! - **Cipher**: Python encrypts with Fernet (AES128-CBC + HMAC) or NaCl secretbox; Rust uses
//!   XChaCha20-Poly1305 with an Argon2id-derived key (Python's exact Argon2 parameters). The
//!   sealed format is `mn1.<b64(nonce ‖ ciphertext)>`. Cross-language *encrypted* payloads are
//!   mutually opaque — both sides already store-and-relay payloads they cannot decrypt
//!   (Python's no-key branch), so plaintext sync interoperates fully and encrypted events relay
//!   without loss.
//! - **Apply-side event logging**: Python's `remember` never writes `memory_events`; Rust's
//!   write path does. `push_changes` suppresses the write-path event log around the apply
//!   (thread-local guard) and then inserts the peer's original event verbatim — net effect
//!   identical to Python (one log row per replicated event, no ping-pong growth).
//! - **Push cursor**: Python's push phase sends local events newer than the *remote's*
//!   high-water timestamp (`/sync/pull` -> `next_cursor`), which permanently masks any local
//!   event older than the remote's newest — bidirectional sync never converges once histories
//!   interleave. Rust keeps a per-remote `last_push_cursor_<url>` over the local log instead;
//!   re-sends stay idempotent through the remote's `event_hash` dedup.
//! - `get_status(remote)` reports the persisted per-remote sync metadata but does not run a
//!   live pull (Python's `get_status(remote_url=...)` embedded a mutating `sync_with` inside a
//!   status read; the tool layer never used that branch).
//! - No env reads: key material and remote URLs are injected via
//!   [`MnemosyneConfig`](crate::MnemosyneConfig) (`sync_remote`/`sync_token`/`sync_key`/
//!   `sync_mode`), resolved by the node's config layer.
//! - **No listener**: Python's `sync_server.py` binds its own HTTP socket; this crate only
//!   ships the outbound client plus the transport-free [`endpoints`] a host may mount. The
//!   node owns transport (see the [`endpoints`] docs).

pub mod endpoints;

use crate::engine::{Engine, RememberArgs};
use crate::error::{Error, Result};
use crate::util;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// The valid event operations (`sync.py` `log_event` L714 + the table CHECK constraint).
pub const SYNC_OPERATIONS: &[&str] = &["CREATE", "UPDATE", "DELETE", "CONSOLIDATE"];

/// Parse sync timestamps consistently (`_parse_sync_timestamp` L25-L34): RFC3339 with a `Z` or
/// numeric offset, falling back to naive ISO treated as UTC (Python `fromisoformat` accepts
/// naive strings).
fn parse_sync_timestamp(value: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(value) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(value, fmt) {
            return Some(naive.and_utc());
        }
    }
    None
}

/// A UUIDv4-format event id from OS randomness (`sync.py` `log_event` L717 `uuid.uuid4()`;
/// no uuid crate — 16 random bytes with the version/variant bits set, hex-grouped 8-4-4-4-12).
fn uuid4() -> String {
    use chacha20poly1305::aead::rand_core::RngCore;
    let mut b = [0u8; 16];
    chacha20poly1305::aead::OsRng.fill_bytes(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

// ── SyncEvent ───────────────────────────────────────────────────────────────────────────────

fn default_parent_ids() -> String {
    "[]".to_string()
}

fn default_importance() -> Option<f64> {
    Some(0.5)
}

/// Accept `parent_event_ids` as either a JSON-encoded string (the storage form) or a bare array
/// (what a peer's `from_dict` tolerates), normalizing to the string form.
fn de_parent_ids<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<String, D::Error> {
    let v = Value::deserialize(d)?;
    Ok(match v {
        Value::String(s) => s,
        Value::Array(_) => v.to_string(),
        Value::Null => default_parent_ids(),
        other => other.to_string(),
    })
}

/// A tracked sync event representing a memory mutation (`SyncEvent`). Field set and JSON names
/// match Python's dataclass; unknown incoming keys are ignored (`from_dict` L61-L65).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncEvent {
    /// Unique event id (UUIDv4 for locally minted events).
    pub event_id: String,
    /// The mutated memory row.
    pub memory_id: String,
    /// `CREATE` | `UPDATE` | `DELETE` | `CONSOLIDATE`.
    pub operation: String,
    /// Mutation time (ISO, UTC).
    pub timestamp: String,
    /// The device that produced the event.
    pub device_id: String,
    /// JSON-encoded (or [`SyncEncryption`]-sealed) mutation payload.
    #[serde(default)]
    pub payload: Option<String>,
    /// JSON-encoded array of causal parent event ids (v2 conflict resolution).
    #[serde(default = "default_parent_ids", deserialize_with = "de_parent_ids")]
    pub parent_event_ids: String,
    /// Row importance at mutation time (`None` survives the wire like Python's dataclass).
    #[serde(default = "default_importance")]
    pub importance: Option<f64>,
    /// Optional expiry for the event itself.
    #[serde(default)]
    pub expiry: Option<String>,
    /// Deterministic dedup hash (`_compute_event_hash`).
    #[serde(default)]
    pub event_hash: Option<String>,
}

impl SyncEvent {
    /// Parse a wire dict, ignoring unknown keys (`from_dict`).
    pub fn from_value(v: &Value) -> Result<Self> {
        Ok(serde_json::from_value(v.clone())?)
    }

    /// The wire dict (`to_dict`).
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("SyncEvent serializes")
    }

    /// The causal parent ids (`ConflictResolution._parse_parent_ids`): JSON list else empty.
    pub fn parent_ids(&self) -> Vec<String> {
        serde_json::from_str::<Vec<String>>(&self.parent_event_ids).unwrap_or_default()
    }
}

// ── SyncEncryption ──────────────────────────────────────────────────────────────────────────

/// Prefix of the Rust sealed-payload format (`mn1.<b64(nonce ‖ ct)>`).
const SEALED_PREFIX: &str = "mn1.";

/// Encryption for sync payloads (`SyncEncryption`): XChaCha20-Poly1305 over an
/// Argon2id-derived or caller-supplied 32-byte key. See the module docs for the format and the
/// cross-language stance.
pub struct SyncEncryption {
    key: [u8; 32],
}

impl SyncEncryption {
    /// Wrap a raw 32-byte key.
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Derive a 32-byte key from a passphrase (`derive_key` L97-L141): Argon2id with Python's
    /// parameters (time_cost=2, memory 19 MiB, parallelism=1). Returns `(key, salt)` — the salt
    /// is random when not provided, so callers store it alongside the ciphertext.
    pub fn derive_key(passphrase: &str, salt: Option<&[u8]>) -> Result<([u8; 32], Vec<u8>)> {
        let salt: Vec<u8> = match salt {
            Some(s) => s.to_vec(),
            None => {
                use chacha20poly1305::aead::rand_core::RngCore;
                let mut s = [0u8; 16];
                chacha20poly1305::aead::OsRng.fill_bytes(&mut s);
                s.to_vec()
            }
        };
        let params = argon2::Params::new(19_456, 2, 1, Some(32))
            .map_err(|e| Error::Invalid(format!("argon2 params: {e}")))?;
        let argon =
            argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
        let mut key = [0u8; 32];
        argon
            .hash_password_into(passphrase.as_bytes(), &salt, &mut key)
            .map_err(|e| Error::Invalid(format!("argon2 derive: {e}")))?;
        Ok((key, salt))
    }

    /// Generate a random 32-byte key, urlsafe-base64 encoded (`generate_key`).
    pub fn generate_key() -> String {
        use chacha20poly1305::aead::rand_core::RngCore;
        let mut key = [0u8; 32];
        chacha20poly1305::aead::OsRng.fill_bytes(&mut key);
        base64::engine::general_purpose::URL_SAFE.encode(key)
    }

    /// Load a key from a source string: a key-file path (optionally `file:`-prefixed) or a raw
    /// urlsafe-base64 key, tolerating stripped padding (`from_config` L196-L231 minus the env
    /// read — key material is injected through config, never read from the process environment).
    pub fn from_key_source(source: &str) -> Result<Option<Self>> {
        let source = source.trim();
        if source.is_empty() {
            return Ok(None);
        }
        let raw = if let Some(path) = source.strip_prefix("file:") {
            std::fs::read_to_string(path)?.trim().to_string()
        } else if std::path::Path::new(source).is_file() {
            std::fs::read_to_string(source)?.trim().to_string()
        } else {
            source.to_string()
        };
        let decoded = base64::engine::general_purpose::URL_SAFE
            .decode(&raw)
            .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&raw))
            .map_err(|_| {
                Error::Invalid(
                    "sync key source is neither a file path nor a valid base64-encoded key"
                        .to_string(),
                )
            })?;
        let key: [u8; 32] = decoded.try_into().map_err(|v: Vec<u8>| {
            Error::Invalid(format!("sync key must be 32 bytes, got {}", v.len()))
        })?;
        Ok(Some(Self::new(key)))
    }

    /// Serialize `payload` to JSON, seal it, and return the transportable string
    /// (`encrypt_payload`).
    pub fn encrypt(&self, payload: &Value) -> String {
        use chacha20poly1305::aead::rand_core::RngCore;
        use chacha20poly1305::aead::{Aead, KeyInit};
        use chacha20poly1305::{XChaCha20Poly1305, XNonce};
        let cipher = XChaCha20Poly1305::new((&self.key).into());
        let mut nonce = [0u8; 24];
        chacha20poly1305::aead::OsRng.fill_bytes(&mut nonce);
        let ct = cipher
            .encrypt(XNonce::from_slice(&nonce), payload.to_string().as_bytes())
            .expect("XChaCha20-Poly1305 encryption is infallible for in-memory buffers");
        let mut sealed = nonce.to_vec();
        sealed.extend_from_slice(&ct);
        format!(
            "{SEALED_PREFIX}{}",
            base64::engine::general_purpose::STANDARD.encode(sealed)
        )
    }

    /// Unseal an encrypted payload back to JSON (`decrypt_payload`).
    pub fn decrypt(&self, encrypted: &str) -> Result<Value> {
        use chacha20poly1305::aead::{Aead, KeyInit};
        use chacha20poly1305::{XChaCha20Poly1305, XNonce};
        let body = encrypted
            .strip_prefix(SEALED_PREFIX)
            .ok_or_else(|| Error::Invalid("not a sealed sync payload".to_string()))?;
        let sealed = base64::engine::general_purpose::STANDARD
            .decode(body)
            .map_err(|e| Error::Invalid(format!("sealed payload base64: {e}")))?;
        if sealed.len() < 24 {
            return Err(Error::Invalid("sealed payload too short".to_string()));
        }
        let (nonce, ct) = sealed.split_at(24);
        let cipher = XChaCha20Poly1305::new((&self.key).into());
        let plain = cipher
            .decrypt(XNonce::from_slice(nonce), ct)
            .map_err(|_| Error::Invalid("sync payload decryption failed".to_string()))?;
        Ok(serde_json::from_slice(&plain)?)
    }

    /// Whether a payload string looks sealed: the Rust `mn1.` format or Python's Fernet
    /// (`gAAAAA` base64 prefix, `push_changes` L944). Sealed payloads a node cannot decrypt are
    /// stored opaque and relayed.
    pub fn is_encrypted(payload: &str) -> bool {
        payload.starts_with(SEALED_PREFIX) || payload.starts_with("gAAAAA")
    }
}

// ── ConflictResolution ──────────────────────────────────────────────────────────────────────

/// v1/v2 conflict resolution (`ConflictResolution`): v1 is last-writer-wins with importance and
/// device-id tiebreakers; v2 consults the `parent_event_ids` causal chain first.
pub struct ConflictResolution;

impl ConflictResolution {
    /// Pick the winning event (`resolve`): latest `timestamp` (string compare, like Python's
    /// tuple sort) -> highest `importance` (`None` = 0.0) -> highest `device_id`. Ties keep the
    /// first event in input order (Python's stable descending sort).
    pub fn resolve(events: &[SyncEvent]) -> Result<&SyncEvent> {
        if events.is_empty() {
            return Err(Error::Invalid("Cannot resolve empty event list".into()));
        }
        // `max_by` returns the LAST maximum; iterating reversed makes ties resolve to the
        // FIRST in original order, matching Python's stable `sorted(reverse=True)[0]`.
        Ok(events
            .iter()
            .rev()
            .max_by(|a, b| {
                (
                    a.timestamp.as_str(),
                    a.importance.unwrap_or(0.0),
                    a.device_id.as_str(),
                )
                    .partial_cmp(&(
                        b.timestamp.as_str(),
                        b.importance.unwrap_or(0.0),
                        b.device_id.as_str(),
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty"))
    }

    /// Resolve using version-chain information (v2, `resolve_with_chain`): if event B lists
    /// event A among its (transitive) `parent_event_ids`, B strictly supersedes A. Events not
    /// dominated by any other survive; a single survivor wins outright, several fall back to
    /// [`Self::resolve`].
    pub fn resolve_with_chain(events: &[SyncEvent]) -> Result<&SyncEvent> {
        if events.is_empty() {
            return Err(Error::Invalid("Cannot resolve empty event list".into()));
        }
        if events.len() == 1 {
            return Ok(&events[0]);
        }
        // Transitive ancestors per event (BFS fixed-point over the in-group parent map).
        let mut ancestors: HashMap<&str, HashSet<String>> = events
            .iter()
            .map(|ev| (ev.event_id.as_str(), ev.parent_ids().into_iter().collect()))
            .collect();
        let ids: Vec<&str> = events.iter().map(|ev| ev.event_id.as_str()).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for id in &ids {
                let current: Vec<String> = ancestors[id].iter().cloned().collect();
                let mut expanded: HashSet<String> = ancestors[id].clone();
                for pid in &current {
                    if let Some(parents) = ancestors.get(pid.as_str()) {
                        expanded.extend(parents.iter().cloned());
                    }
                }
                if expanded.len() != ancestors[id].len() {
                    ancestors.insert(id, expanded);
                    changed = true;
                }
            }
        }
        let dominated: HashSet<&str> = events
            .iter()
            .filter(|a| {
                events.iter().any(|b| {
                    b.event_id != a.event_id
                        && ancestors
                            .get(b.event_id.as_str())
                            .is_some_and(|set| set.contains(&a.event_id))
                })
            })
            .map(|a| a.event_id.as_str())
            .collect();
        let undominated: Vec<SyncEvent> = events
            .iter()
            .filter(|ev| !dominated.contains(ev.event_id.as_str()))
            .cloned()
            .collect();
        if undominated.len() == 1 {
            let id = &undominated[0].event_id;
            return Ok(events.iter().find(|ev| &ev.event_id == id).expect("member"));
        }
        let winner_id = Self::resolve(&undominated)?.event_id.clone();
        Ok(events
            .iter()
            .find(|ev| ev.event_id == winner_id)
            .expect("member"))
    }

    /// Find groups of conflicting events (`detect_conflicts`): same `memory_id`, timestamps
    /// within `window_seconds` of each other, at least one event from each side.
    pub fn detect_conflicts(
        local_events: &[SyncEvent],
        remote_events: &[SyncEvent],
        window_seconds: f64,
    ) -> Vec<Vec<SyncEvent>> {
        let mut local_by_mid: HashMap<&str, Vec<&SyncEvent>> = HashMap::new();
        for ev in local_events {
            local_by_mid.entry(&ev.memory_id).or_default().push(ev);
        }
        let mut remote_by_mid: HashMap<&str, Vec<&SyncEvent>> = HashMap::new();
        for ev in remote_events {
            remote_by_mid.entry(&ev.memory_id).or_default().push(ev);
        }
        let mut mids: Vec<&str> = local_by_mid
            .keys()
            .chain(remote_by_mid.keys())
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        mids.sort_unstable(); // deterministic group order (Python set order is arbitrary)

        let mut conflicts = Vec::new();
        for mid in mids {
            let (Some(locals), Some(remotes)) = (local_by_mid.get(mid), remote_by_mid.get(mid))
            else {
                continue;
            };
            for lev in locals {
                let Some(lts) = parse_sync_timestamp(&lev.timestamp) else {
                    continue;
                };
                for rev in remotes {
                    let Some(rts) = parse_sync_timestamp(&rev.timestamp) else {
                        continue;
                    };
                    let diff = (lts - rts).num_milliseconds().abs() as f64 / 1000.0;
                    if diff > window_seconds {
                        continue;
                    }
                    // Group = this pair + any other remote events in window of the local one.
                    let mut group: Vec<SyncEvent> = vec![(*lev).clone(), (*rev).clone()];
                    for rev2 in remotes {
                        if rev2.event_id == rev.event_id {
                            continue;
                        }
                        if let Some(rts2) = parse_sync_timestamp(&rev2.timestamp) {
                            let diff2 = (lts - rts2).num_milliseconds().abs() as f64 / 1000.0;
                            if diff2 <= window_seconds {
                                group.push((*rev2).clone());
                            }
                        }
                    }
                    let mut seen = HashSet::new();
                    group.retain(|ev| seen.insert(ev.event_id.clone()));
                    if group.len() > 1 {
                        conflicts.push(group);
                    }
                }
            }
        }
        conflicts
    }

    /// Build merge-proposal structures for LLM consumption (`propose_merge`): one proposal per
    /// conflict group with candidate summaries, a heuristic winner (highest importance, then
    /// timestamp), and the caller's context. Calls no LLM itself — an agent layer consumes it.
    pub fn propose_merge(
        conflict_groups: &[Vec<SyncEvent>],
        full_context: Option<&Value>,
    ) -> Vec<Value> {
        let mut proposals = Vec::new();
        for group in conflict_groups {
            if group.len() < 2 {
                continue;
            }
            let candidates: Vec<Value> = group
                .iter()
                .map(|ev| {
                    let content = ev
                        .payload
                        .as_deref()
                        .and_then(|p| serde_json::from_str::<Value>(p).ok())
                        .and_then(|p| {
                            p.get("content")
                                .and_then(|c| c.as_str())
                                .map(str::to_string)
                        })
                        .unwrap_or_default();
                    json!({
                        "device": ev.device_id,
                        "content": content,
                        "importance": ev.importance.unwrap_or(0.5),
                        "timestamp": ev.timestamp,
                        "event_id": ev.event_id,
                    })
                })
                .collect();
            let best_idx = (0..candidates.len())
                .max_by(|&i, &j| {
                    let key = |k: usize| {
                        (
                            candidates[k]["importance"].as_f64().unwrap_or(0.5),
                            candidates[k]["timestamp"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                        )
                    };
                    key(i)
                        .partial_cmp(&key(j))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            proposals.push(json!({
                "memory_id": group[0].memory_id,
                "candidates": candidates,
                "suggested_action": "keep_latest",
                "suggested_winner_index": best_idx,
                "context": full_context.cloned().unwrap_or_else(|| json!({})),
            }));
        }
        proposals
    }
}

// ── SyncEngine ──────────────────────────────────────────────────────────────────────────────

/// Orchestrates memory synchronization between banks (`SyncEngine`), over the `memory_events`
/// log + `sync_meta` state the schema already carries.
pub struct SyncEngine<'e> {
    engine: &'e Engine,
    /// Stable device identity for minted events (explicit > `sync_meta` > generated).
    pub device_id: String,
    encryption: Option<SyncEncryption>,
}

impl<'e> SyncEngine<'e> {
    /// A sync engine over `engine`'s bank (`__init__` L588-L638). `device_id`: explicit wins;
    /// otherwise the bank's persisted identity loads (or mints + persists) via `sync_meta`.
    pub fn new(
        engine: &'e Engine,
        device_id: Option<String>,
        encryption: Option<SyncEncryption>,
    ) -> Result<Self> {
        let device_id = match device_id {
            Some(id) => id,
            None => engine.with_conn(|conn| Ok(engine.device_id(conn)))?,
        };
        Ok(Self {
            engine,
            device_id,
            encryption,
        })
    }

    /// Read a `sync_meta` value (`_meta_get`).
    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        self.engine.with_conn(|conn| {
            use rusqlite::OptionalExtension;
            Ok(conn
                .query_row("SELECT value FROM sync_meta WHERE key = ?1", [key], |r| {
                    r.get(0)
                })
                .optional()?)
        })
    }

    /// Write a `sync_meta` value (`_meta_set`).
    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        self.engine.with_conn(|conn| {
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?1, ?2)",
                [key, value],
            )?;
            Ok(())
        })
    }

    /// Deterministic dedup hash for an event (`_compute_event_hash` L692-L699), with Python
    /// float formatting for `importance` (`0.5` -> `"0.5"`; `None` -> `"None"`).
    pub fn compute_event_hash(event: &SyncEvent) -> String {
        use sha2::{Digest, Sha256};
        let importance = event
            .importance
            .map(util::py_float)
            .unwrap_or_else(|| "None".to_string());
        let preimage = format!(
            "{}|{}|{}|{}|{}|{}|{importance}",
            event.memory_id,
            event.operation,
            event.timestamp,
            event.device_id,
            event.payload.as_deref().unwrap_or(""),
            event.parent_event_ids,
        );
        let mut h = Sha256::new();
        h.update(preimage.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// Create and persist a sync event (`log_event`): the primary seam for recording mutations
    /// to replicate. Payloads seal when encryption is configured.
    pub fn log_event(
        &self,
        memory_id: &str,
        operation: &str,
        payload: Option<&Value>,
        importance: f64,
        parent_event_ids: Option<&[String]>,
    ) -> Result<SyncEvent> {
        if !SYNC_OPERATIONS.contains(&operation) {
            return Err(Error::Invalid(format!("Invalid operation: {operation:?}")));
        }
        let payload_str = payload.map(|p| match &self.encryption {
            Some(enc) => enc.encrypt(p),
            None => p.to_string(),
        });
        let mut event = SyncEvent {
            event_id: uuid4(),
            memory_id: memory_id.to_string(),
            operation: operation.to_string(),
            timestamp: util::now_iso(),
            device_id: self.device_id.clone(),
            payload: payload_str,
            parent_event_ids: serde_json::to_string(&parent_event_ids.unwrap_or(&[]))?,
            importance: Some(importance),
            expiry: None,
            event_hash: None,
        };
        event.event_hash = Some(Self::compute_event_hash(&event));
        self.engine.with_conn(|conn| {
            conn.execute(
                "INSERT INTO memory_events \
                 (event_id, memory_id, operation, timestamp, device_id, payload, \
                  parent_event_ids, importance, expiry, event_hash) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    event.event_id,
                    event.memory_id,
                    event.operation,
                    event.timestamp,
                    event.device_id,
                    event.payload,
                    event.parent_event_ids,
                    event.importance,
                    event.expiry,
                    event.event_hash,
                ],
            )?;
            Ok(())
        })?;
        Ok(event)
    }

    /// Backfill CREATE events for working rows not yet in the event log
    /// (`_find_unlogged_memories`). Rust's write path event-logs every mutation, so this only
    /// finds rows written by other implementations (a Python bank) or pre-sync history.
    pub fn find_unlogged_memories(&self, limit: usize) -> Result<Vec<SyncEvent>> {
        struct Row {
            id: String,
            content: String,
            source: String,
            importance: f64,
            metadata_json: Option<String>,
        }
        let unlogged: Vec<Row> = self.engine.with_conn(|conn| {
            let logged: HashSet<String> = {
                let mut stmt = conn.prepare("SELECT DISTINCT memory_id FROM memory_events")?;
                let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
                rows.collect::<std::result::Result<_, _>>()?
            };
            let mut stmt = conn.prepare(
                "SELECT id, content, COALESCE(source, 'conversation'), \
                        COALESCE(importance, 0.5), metadata_json \
                 FROM main.working_memory ORDER BY timestamp ASC LIMIT ?1",
            )?;
            let rows = stmt.query_map([limit as i64], |r| {
                Ok(Row {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get(2)?,
                    importance: r.get(3)?,
                    metadata_json: r.get(4)?,
                })
            })?;
            Ok(rows
                .filter_map(std::result::Result::ok)
                .filter(|row| !logged.contains(&row.id))
                .collect())
        })?;

        let mut created = Vec::new();
        for row in unlogged {
            let mut payload = json!({"content": row.content, "source": row.source});
            if let Some(meta) = &row.metadata_json {
                payload["metadata_json"] = serde_json::from_str::<Value>(meta)
                    .unwrap_or_else(|_| Value::String(meta.clone()));
            }
            created.push(self.log_event(
                &row.id,
                "CREATE",
                Some(&payload),
                row.importance,
                None,
            )?);
        }
        Ok(created)
    }

    /// Pull events from the local log since a cursor (`pull_changes`): ordered
    /// `(timestamp, event_id)` ascending, `limit`-paged. Returns
    /// `{events, next_cursor, has_more, total}`.
    pub fn pull_changes(&self, since_cursor: Option<&str>, limit: usize) -> Result<Value> {
        let events: Vec<SyncEvent> = self.engine.with_conn(|conn| {
            // `timestamp > ''` is vacuously true, so an absent cursor selects everything.
            let mut stmt = conn.prepare(
                "SELECT event_id, memory_id, operation, timestamp, device_id, payload, \
                        parent_event_ids, importance, expiry, event_hash \
                 FROM memory_events WHERE timestamp > ?1 \
                 ORDER BY timestamp ASC, event_id ASC LIMIT ?2",
            )?;
            let params = rusqlite::params![since_cursor.unwrap_or(""), (limit + 1) as i64];
            let rows = stmt.query_map(params, |r| {
                Ok(SyncEvent {
                    event_id: r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    memory_id: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    operation: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    device_id: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    payload: r.get(5)?,
                    parent_event_ids: r
                        .get::<_, Option<String>>(6)?
                        .unwrap_or_else(default_parent_ids),
                    importance: r.get(7)?,
                    expiry: r.get(8)?,
                    event_hash: r.get(9)?,
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Error::from)
        })?;

        let has_more = events.len() > limit;
        let events = &events[..events.len().min(limit)];
        let next_cursor = events.last().map(|ev| ev.timestamp.clone());
        Ok(json!({
            "events": events.iter().map(SyncEvent::to_value).collect::<Vec<_>>(),
            "next_cursor": next_cursor,
            "has_more": has_more,
            "total": events.len(),
        }))
    }

    /// Validate, deduplicate, conflict-resolve, and apply incoming events (`push_changes`).
    /// Mutations route through the full write pipeline — [`Engine::remember`] with the peer's
    /// `memory_id` (FTS, dedup, knowledge ingestion), [`Engine::forget`] for DELETE — then the
    /// peer's original event lands in the log with `synced_at` set (idempotent via
    /// `INSERT OR IGNORE` + `event_hash` dedup). Returns
    /// `{accepted, duplicates, conflicts, errors, details}`.
    pub fn push_changes(&self, events: &[Value]) -> Result<Value> {
        let (mut accepted, mut duplicates, mut conflicts, mut errors) = (0u64, 0u64, 0u64, 0u64);
        let mut details: Vec<String> = Vec::new();

        let known_hashes: HashSet<String> = self.engine.with_conn(|conn| {
            let mut stmt =
                conn.prepare("SELECT event_hash FROM memory_events WHERE event_hash IS NOT NULL")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<std::result::Result<_, _>>()
                .map_err(Error::from)
        })?;

        let mut incoming: Vec<SyncEvent> = Vec::new();
        for raw in events {
            match SyncEvent::from_value(raw) {
                Ok(ev) => {
                    if ev
                        .event_hash
                        .as_deref()
                        .is_some_and(|h| known_hashes.contains(h))
                    {
                        duplicates += 1;
                        continue;
                    }
                    incoming.push(ev);
                }
                Err(e) => {
                    errors += 1;
                    details.push(format!("invalid event: {e}"));
                }
            }
        }

        // Conflict detection against the local log (window ±5s), v1 LWW resolution; losing
        // *incoming* events are dropped (a losing local event stays — parity with Python).
        if !incoming.is_empty() {
            let local_raw = self.pull_changes(None, 5000)?;
            let local_events: Vec<SyncEvent> = local_raw["events"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| SyncEvent::from_value(v).ok())
                        .collect()
                })
                .unwrap_or_default();
            let groups = ConflictResolution::detect_conflicts(&local_events, &incoming, 5.0);
            let mut losing_ids: HashSet<String> = HashSet::new();
            for group in &groups {
                let winner_id = ConflictResolution::resolve(group)?.event_id.clone();
                for ev in group {
                    if ev.event_id != winner_id {
                        losing_ids.insert(ev.event_id.clone());
                    }
                }
                conflicts += (group.len() - 1) as u64;
            }
            incoming.retain(|ev| !losing_ids.contains(&ev.event_id));
        }

        for ev in incoming {
            match self.apply_event(&ev) {
                Ok(()) => accepted += 1,
                Err(e) => {
                    errors += 1;
                    details.push(format!("event {}: {e}", ev.event_id));
                    tracing::warn!(event_id = %ev.event_id, error = %e, "failed to apply sync event");
                }
            }
        }

        Ok(json!({
            "accepted": accepted,
            "duplicates": duplicates,
            "conflicts": conflicts,
            "errors": errors,
            "details": details,
        }))
    }

    /// Apply one incoming event (`push_changes` body L939-L1038): decode the payload, route the
    /// mutation through the full pipeline, and record the event with `synced_at`.
    fn apply_event(&self, ev: &SyncEvent) -> Result<()> {
        // Decode the payload: sealed payloads decrypt when we hold the key; without a key they
        // stay opaque (the event still logs, so it relays to better-equipped peers). A failed
        // decrypt with a key, or unparseable plaintext, is an error (the event does not log) —
        // Python's try/except placement.
        let payload_dict: Option<Value> = match &ev.payload {
            Some(p) if SyncEncryption::is_encrypted(p) => match &self.encryption {
                Some(enc) => Some(enc.decrypt(p)?),
                None => None,
            },
            Some(p) => Some(serde_json::from_str(p)?),
            None => None,
        };

        let mut content = String::new();
        let mut source = "sync".to_string();
        let mut importance = ev.importance.unwrap_or(0.5);
        let mut metadata: Option<Value> = None;
        if let Some(p) = &payload_dict {
            content = p
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            if let Some(s) = p.get("source").and_then(|v| v.as_str()) {
                source = s.to_string();
            }
            if let Some(i) = p.get("importance").and_then(|v| v.as_f64()) {
                importance = i;
            }
            metadata = match p.get("metadata_json") {
                Some(Value::String(s)) => serde_json::from_str(s).ok(),
                Some(v) if !v.is_null() => Some(v.clone()),
                _ => None,
            };
        }

        match ev.operation.as_str() {
            "DELETE" => {
                self.engine.forget(&ev.memory_id)?;
            }
            "CREATE" | "UPDATE" | "CONSOLIDATE" if !content.is_empty() => {
                // Suppress the write path's own event logging: the peer's original event is
                // the record of this mutation (see the module docs).
                let _guard = crate::engine::suppress_event_log();
                self.engine.remember(
                    &content,
                    &RememberArgs {
                        source,
                        importance,
                        metadata,
                        memory_id: Some(ev.memory_id.clone()),
                        ..RememberArgs::default()
                    },
                )?;
            }
            _ => {} // no content — just log the event
        }

        self.engine.with_conn(|conn| {
            conn.execute(
                "INSERT OR IGNORE INTO memory_events \
                 (event_id, memory_id, operation, timestamp, device_id, payload, \
                  parent_event_ids, importance, expiry, event_hash, synced_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    ev.event_id,
                    ev.memory_id,
                    ev.operation,
                    ev.timestamp,
                    ev.device_id,
                    ev.payload,
                    ev.parent_event_ids,
                    ev.importance,
                    ev.expiry,
                    ev.event_hash,
                    util::now_iso(),
                ],
            )?;
            Ok(())
        })
    }

    /// POST JSON to a sync server endpoint; never raises (`sync_adapter.py` `_http_post`).
    /// HTTP-error bodies pass through when they parse as JSON, else an error shape returns.
    pub async fn http_post(
        remote_url: &str,
        path: &str,
        body: &Value,
        api_key: Option<&str>,
    ) -> Value {
        let url = format!("{}{path}", remote_url.trim_end_matches('/'));
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => return json!({"status": "error", "error": e.to_string()}),
        };
        let mut req = client.post(&url).json(body);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                match resp.json::<Value>().await {
                    Ok(v) => v,
                    Err(_) => json!({
                        "status": "error",
                        "error": format!("HTTP {}: unparseable response", status.as_u16()),
                    }),
                }
            }
            Err(e) => json!({"status": "error", "error": format!("{path}: {e}")}),
        }
    }

    /// Run a full sync cycle with a remote server (`sync_with`): `mode` is `push`, `pull`, or
    /// `bidirectional`. Returns `{remote, mode, push, pull, errors}`.
    pub async fn sync_with(&self, remote_url: &str, mode: &str, api_key: Option<&str>) -> Value {
        let mut result = json!({
            "remote": remote_url,
            "mode": mode,
            "push": Value::Null,
            "pull": Value::Null,
            "errors": [],
        });
        let push_err = |result: &mut Value, msg: String| {
            result["errors"]
                .as_array_mut()
                .expect("array")
                .push(json!(msg));
        };

        // Phase 1: push local changes to the remote.
        if mode == "push" || mode == "bidirectional" {
            match self.find_unlogged_memories(5000) {
                Ok(new_events) if !new_events.is_empty() => {
                    tracing::debug!(count = new_events.len(), "found unlogged memories to sync");
                }
                Ok(_) => {}
                Err(e) => push_err(&mut result, format!("find_unlogged: {e}")),
            }
            // Send everything past the persisted per-remote push cursor. (Python instead asked
            // the remote for its high-water `next_cursor` and filtered the local log by it —
            // masking every local event older than the remote's newest; see the module docs.)
            let cursor_key = format!("last_push_cursor_{remote_url}");
            let since = self.meta_get(&cursor_key).ok().flatten();
            match self.pull_changes(since.as_deref(), 5000) {
                Ok(local_changes) => {
                    let events = local_changes["events"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default();
                    if events.is_empty() {
                        result["push"] = json!({"accepted": 0, "duplicates": 0, "conflicts": 0});
                    } else {
                        let resp = Self::http_post(
                            remote_url,
                            "/sync/push",
                            &json!({"events": events, "device_id": self.device_id}),
                            api_key,
                        )
                        .await;
                        if resp["status"] == "error" {
                            push_err(
                                &mut result,
                                format!("/sync/push: {}", resp["error"].as_str().unwrap_or("?")),
                            );
                        } else if let Some(next) =
                            local_changes.get("next_cursor").and_then(|v| v.as_str())
                        {
                            let _ = self.meta_set(&cursor_key, next);
                        }
                        result["push"] = resp;
                    }
                }
                Err(e) => push_err(&mut result, format!("pull_changes: {e}")),
            }
        }

        // Phase 2: pull remote changes and apply locally.
        if mode == "pull" || mode == "bidirectional" {
            let cursor_key = format!("last_sync_cursor_{remote_url}");
            let since_cursor = match self.meta_get(&cursor_key) {
                Ok(Some(c)) => Some(c),
                _ => self
                    .engine
                    .with_conn(|conn| {
                        Ok(conn.query_row(
                            "SELECT MAX(timestamp) FROM memory_events WHERE device_id != ?1",
                            [&self.device_id],
                            |r| r.get::<_, Option<String>>(0),
                        )?)
                    })
                    .ok()
                    .flatten(),
            };
            let pull_resp = Self::http_post(
                remote_url,
                "/sync/pull",
                &json!({"since": since_cursor, "device_id": self.device_id, "limit": 5000}),
                api_key,
            )
            .await;
            let events = pull_resp
                .get("events")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if events.is_empty() {
                result["pull"] = json!({"events_fetched": 0});
            } else {
                match self.push_changes(&events) {
                    Ok(stats) => {
                        result["pull"] = json!({
                            "events_fetched": events.len(),
                            "accepted": stats["accepted"],
                            "duplicates": stats["duplicates"],
                            "conflicts": stats["conflicts"],
                            "errors": stats["errors"],
                        });
                        if let Some(next) = pull_resp.get("next_cursor").and_then(|v| v.as_str()) {
                            let _ = self.meta_set(&cursor_key, next);
                        }
                        if stats["accepted"].as_u64().unwrap_or(0) > 0 {
                            let _ = self
                                .meta_set(&format!("last_sync_at_{remote_url}"), &util::now_iso());
                        }
                    }
                    Err(e) => push_err(&mut result, format!("apply pulled events: {e}")),
                }
            }
        }

        result
    }

    /// Local sync status and statistics (`get_status`): event/device counts, operation
    /// breakdown, synced count, plus the persisted per-remote sync metadata when `remote_url`
    /// is given (read-only — no live probe; see the module docs).
    pub fn get_status(&self, remote_url: Option<&str>) -> Result<Value> {
        let (total, devices, last_event, synced) = self.engine.with_conn(|conn| {
            Ok((
                conn.query_row("SELECT COUNT(*) FROM memory_events", [], |r| {
                    r.get::<_, i64>(0)
                })?,
                conn.query_row(
                    "SELECT COUNT(DISTINCT device_id) FROM memory_events",
                    [],
                    |r| r.get::<_, i64>(0),
                )?,
                conn.query_row("SELECT MAX(timestamp) FROM memory_events", [], |r| {
                    r.get::<_, Option<String>>(0)
                })?,
                conn.query_row(
                    "SELECT COUNT(*) FROM memory_events WHERE synced_at IS NOT NULL",
                    [],
                    |r| r.get::<_, i64>(0),
                )?,
            ))
        })?;
        let breakdown: Vec<(String, i64)> = self.engine.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT operation, COUNT(*) AS cnt FROM memory_events \
                 GROUP BY operation ORDER BY cnt DESC",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<std::result::Result<_, _>>()
                .map_err(Error::from)
        })?;

        let mut breakdown_map = serde_json::Map::new();
        for (op, cnt) in breakdown {
            breakdown_map.insert(op, json!(cnt));
        }
        let mut status = json!({
            "device_id": self.device_id,
            "total_events": total,
            "device_count": devices,
            "last_event_time": last_event,
            "operation_breakdown": breakdown_map,
            "synced_events": synced,
        });
        if let Some(remote) = remote_url {
            status["remote"] = json!(remote);
            if let Some(last) = self.meta_get(&format!("last_sync_at_{remote}"))? {
                status["last_sync"] = json!(last);
            }
        }
        Ok(status)
    }
}

#[cfg(test)]
mod tests;
