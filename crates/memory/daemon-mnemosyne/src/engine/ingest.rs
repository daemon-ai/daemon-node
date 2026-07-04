// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Write path for the BEAM [`Engine`]: `remember`/`remember_with_vector` with exact-content dedup,
//! trust-tier derivation, working-memory trim, deterministic + LLM knowledge ingestion, proactive
//! linking, and the `memory_events` event log. Port of `beam.py` `remember` L2836-L3043.

use super::*;
use crate::dynamics::typed_memory;
use crate::knowledge::{annotations, entities, episodic_graph, temporal, veracity};
use crate::util::py_float;
use crate::{memoria, sanitize, util};
use rusqlite::{params, Connection, OptionalExtension};

/// Ingestion-source -> trust-tier policy map (`beam.py` `TRUST_TIER_MAP` L152-L166). Callers
/// describe WHAT they are (`source`); the engine decides HOW to trust it.
const TRUST_TIER_MAP: &[(&str, &str)] = &[
    ("conversation", "STATED"),
    ("user", "STATED"),
    ("cli", "STATED"),
    ("mcp", "EXTERNAL_WRITE"),
    ("import", "IMPORTED"),
    ("mem0", "IMPORTED"),
    ("honcho_import", "IMPORTED"),
    ("honcho_summary", "IMPORTED"),
    ("consolidation", "DERIVED"),
    ("sleep_consolidation", "DERIVED"),
    ("regex", "DERIVED"),
    ("extraction", "DERIVED"),
    ("unknown", "STATED"),
];

/// The clamp allowlist for explicit trust tiers (`beam.py` L2887).
const TRUST_TIERS: &[&str] = &["STATED", "DERIVED", "EXTERNAL_WRITE", "IMPORTED"];

/// Map an ingestion source to a trust tier (`beam.py` `_source_to_trust_tier` L168-L188): direct
/// map hit first, then the `import`/`mcp` substring heuristics, else the conservative `STATED`.
pub(crate) fn source_to_trust_tier(source: &str) -> &'static str {
    if source.is_empty() {
        return "STATED";
    }
    if let Some((_, tier)) = TRUST_TIER_MAP.iter().find(|(s, _)| *s == source) {
        return tier;
    }
    let lower = source.to_lowercase();
    if lower.contains("import") {
        return "IMPORTED";
    }
    if lower.contains("mcp") {
        return "EXTERNAL_WRITE";
    }
    "STATED"
}

thread_local! {
    /// Set while sync-apply routes a peer mutation through the write pipeline (`sync.rs`
    /// `push_changes`). Python's `remember` never writes `memory_events` (only
    /// `SyncEngine.log_event` does); Rust's write path does — without suppression, applying a
    /// peer's CREATE would mint a second local event for the same row and every sync cycle
    /// would ping-pong-grow both peers' logs. Thread-local so concurrent writers on other
    /// threads keep logging normally.
    static SUPPRESS_EVENT_LOG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII guard suppressing `memory_events` inserts on the current thread (the sync-apply path).
/// The in-process [`crate::streaming::MemoryStream`] still fires — parity with Python, where
/// sync-applied remembers emit stream events but never self-log.
#[cfg_attr(not(feature = "sync"), allow(dead_code))]
pub(crate) struct EventLogSuppressGuard {
    prev: bool,
}

#[cfg_attr(not(feature = "sync"), allow(dead_code))]
pub(crate) fn suppress_event_log() -> EventLogSuppressGuard {
    let prev = SUPPRESS_EVENT_LOG.with(|c| c.replace(true));
    EventLogSuppressGuard { prev }
}

impl Drop for EventLogSuppressGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        SUPPRESS_EVENT_LOG.with(|c| c.set(prev));
    }
}

/// The FTS keyword stop-list used by proactive similarity linking (`beam.py` `_proactively_link`
/// L3382-L3398 — its own inline set, distinct from the entities/synonyms lists).
const LINK_STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "about", "which", "who", "what",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "and", "but", "or", "if", "while",
];

impl Engine {
    /// Store a memory in the working tier (`beam.py` `remember` L2836), keyword-only (no vector).
    /// Equivalent to [`Engine::remember_with_vector`] with no embedding.
    pub fn remember(&self, content: &str, args: &RememberArgs) -> Result<String> {
        self.remember_with_vector(content, args, None, "")
    }

    /// Store a memory in the working tier, optionally persisting a precomputed embedding into
    /// `memory_embeddings` (the f32-BLOB-as-JSON fallback store).
    ///
    /// The embedding is computed by the caller (the async [`MnemosyneProvider`] hooks) and passed
    /// in, so the synchronous engine never blocks on a model call. Ordered steps mirror the Python
    /// write path verbatim: clamp -> sanitize -> trust tier -> classify -> dedup -> insert -> trim
    /// -> embed -> temporal -> entities -> MEMORIA -> graph/veracity -> event -> cache invalidate.
    pub fn remember_with_vector(
        &self,
        content: &str,
        args: &RememberArgs,
        vector: Option<&[f32]>,
        model: &str,
    ) -> Result<String> {
        // Clamp veracity at the lowest-level public ingest path (`beam.py` L2872).
        let veracity_label = veracity::clamp_veracity(&args.veracity, "remember");
        // Sanitize (`beam.py` L2874-L2880): binary/oversized/high-entropy payloads spill to the
        // blob store, leaving a placeholder + `_blob` metadata merged into the caller's metadata.
        let (content, blob_meta) = sanitize::sanitize_content(content, &self.config.blob_dir());
        let mut metadata = match &args.metadata {
            Some(serde_json::Value::Object(m)) => m.clone(),
            _ => serde_json::Map::new(),
        };
        if !blob_meta.as_object().map(|m| m.is_empty()).unwrap_or(true) {
            metadata.insert("_blob".to_string(), blob_meta);
        }
        let metadata_json = serde_json::to_string(&serde_json::Value::Object(metadata))?;
        // Trust tier: explicit arg (clamped) else derived from source (`beam.py` L2883-L2887).
        let trust_tier = match args.trust_tier.as_deref() {
            Some(t) if TRUST_TIERS.contains(&t) => t,
            Some(_) => "STATED",
            None => source_to_trust_tier(&args.source),
        };
        let memory_type = typed_memory::classify(&content).as_str();
        let now = util::now_iso();

        let conn = self.store.conn.lock().unwrap();

        // ── Deduplication: exact (session, content) match (`beam.py` L2911-L2965) ──
        if let Some(existing_id) = self.find_duplicate(&conn, &content)? {
            let channel_id = self
                .config
                .channel_id
                .clone()
                .unwrap_or_else(|| self.config.session_id.clone());
            // The dedup-update clears `consolidated_at` so a re-remembered row becomes eligible
            // for sleep again, and only upgrades veracity when the new label is non-`unknown`.
            conn.execute(
                "UPDATE working_memory \
                 SET importance = MAX(importance, ?1), timestamp = ?2, source = ?3, \
                     valid_until = COALESCE(?4, valid_until), \
                     scope = COALESCE(?5, scope), \
                     author_id = COALESCE(?6, author_id), \
                     author_type = COALESCE(?7, author_type), \
                     channel_id = COALESCE(?8, channel_id), \
                     memory_type = COALESCE(?9, memory_type), \
                     veracity = CASE WHEN ?10 != 'unknown' THEN ?10 ELSE veracity END, \
                     trust_tier = COALESCE(?11, trust_tier), \
                     consolidated_at = NULL, \
                     consolidation_claimed_at = NULL \
                 WHERE id = ?12 AND session_id = ?13",
                params![
                    args.importance,
                    now,
                    args.source,
                    args.valid_until,
                    args.scope,
                    self.config.author_id,
                    self.config.author_type,
                    channel_id,
                    memory_type,
                    veracity_label,
                    trust_tier,
                    existing_id,
                    self.config.session_id,
                ],
            )?;
            if let Some(vector) = vector {
                self.store_embedding(&conn, &existing_id, vector, model)?;
            }
            // Re-run the extraction pipeline the new-row path runs, so duplicate-content writes
            // still populate the knowledge tables (`beam.py` L2939-L2960).
            if args.extract_entities {
                self.extract_and_store_entities(&conn, &existing_id, &content);
            }
            if let Err(e) = memoria::extract_and_store(
                &conn,
                &self.config.session_id,
                &content,
                0,
                &existing_id,
            ) {
                tracing::debug!(error = %e, "memoria extraction failed (non-fatal)");
            }
            self.ingest_graph_and_veracity(&conn, &existing_id, &content, &veracity_label);
            self.emit_event(
                &conn,
                "UPDATE",
                &existing_id,
                Some(&content),
                &args.source,
                args.importance,
            );
            self.audit(&conn, "remember", Some(&existing_id), Some("dedup-update"));
            if let Some(cache) = self.query_cache.get() {
                cache.invalidate();
            }
            if let Some(pm) = self.plugins_if_active() {
                pm.notify_remember(&serde_json::json!({"id": existing_id, "content": content}));
            }
            return Ok(existing_id);
        }

        // ── New row (`beam.py` L2967-L3043) ──
        let id = args
            .memory_id
            .clone()
            .unwrap_or_else(|| util::generate_id(&content));
        // Deterministic temporal extraction (`temporal_parser.py`): populate the event_date
        // columns so recall + degradation can reason over when an event occurred. Python inserts
        // then updates them (L2999-L3017); folding into the INSERT is equivalent.
        let temporal = temporal::extract_temporal(&content);
        let temporal_tags_json = serde_json::to_string(&temporal.temporal_tags)?;
        // Multi-agent identity columns (`beam.py` ctor L2616-L2618 / write L2974): stamp the row
        // with the configured author and the channel (defaulting to the session).
        let channel_id = self
            .config
            .channel_id
            .clone()
            .unwrap_or_else(|| self.config.session_id.clone());
        conn.execute(
            "INSERT INTO working_memory \
             (id, content, source, timestamp, session_id, importance, metadata_json, valid_until, \
              scope, author_id, author_type, channel_id, veracity, memory_type, trust_tier, \
              event_date, event_date_precision, temporal_tags) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                id,
                content,
                args.source,
                now,
                self.config.session_id,
                args.importance,
                metadata_json,
                args.valid_until,
                args.scope,
                self.config.author_id,
                self.config.author_type,
                channel_id,
                veracity_label,
                memory_type,
                trust_tier,
                temporal.event_date,
                temporal.event_date_precision,
                temporal_tags_json,
            ],
        )?;
        self.trim_working_memory(&conn)?;
        if let Some(vector) = vector {
            self.store_embedding(&conn, &id, vector, model)?;
        }
        // Auto temporal annotations (`beam.py` `_add_temporal_triple` L3471-L3494).
        self.add_temporal_triple(&conn, &id, &now, &args.source);
        if args.extract_entities {
            self.extract_and_store_entities(&conn, &id, &content);
        }
        // Always-on MEMORIA regex extraction (`beam.py` L3027-L3033): populate the specialist
        // tables for structured retrieval. Best-effort — extraction must never block storage.
        if let Err(e) = memoria::extract_and_store(&conn, &self.config.session_id, &content, 0, &id)
        {
            tracing::debug!(error = %e, "memoria extraction failed (non-fatal)");
        }
        self.ingest_graph_and_veracity(&conn, &id, &content, &veracity_label);
        self.emit_event(
            &conn,
            "CREATE",
            &id,
            Some(&content),
            &args.source,
            args.importance,
        );
        self.audit(&conn, "remember", Some(&id), None);
        // Invalidate the enhanced-recall query cache (`beam.py` L3041-L3043). Only touched when the
        // cache was actually opened (enhanced recall used), so Base mode pays nothing.
        if let Some(cache) = self.query_cache.get() {
            cache.invalidate();
        }
        if let Some(pm) = self.plugins_if_active() {
            pm.notify_remember(&serde_json::json!({"id": id, "content": content}));
        }
        Ok(id)
    }

    /// Public exact-dedup probe for the tool layer (`__init__.py` `_handle_shared_remember` L1995
    /// checks for an existing row to report `existing_shared` vs `stored_shared`).
    pub fn find_existing(&self, content: &str) -> Result<Option<String>> {
        let conn = self.store.conn.lock().unwrap();
        self.find_duplicate(&conn, content)
    }

    /// Exact same-session content dedup lookup (`beam.py` `_find_duplicate` L2801-L2811).
    fn find_duplicate(&self, conn: &Connection, content: &str) -> Result<Option<String>> {
        Ok(conn
            .query_row(
                "SELECT id FROM working_memory WHERE session_id = ?1 AND content = ?2 LIMIT 1",
                params![self.config.session_id, content],
                |r| r.get::<_, String>(0),
            )
            .optional()?)
    }

    /// Persist a dense embedding for a memory (`beam.py` `_store_working_embedding` L1856).
    pub(crate) fn store_embedding(
        &self,
        conn: &Connection,
        memory_id: &str,
        vector: &[f32],
        model: &str,
    ) -> Result<()> {
        let embedding_json = serde_json::to_string(vector)?;
        conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) \
             VALUES (?1, ?2, ?3)",
            params![memory_id, embedding_json, model],
        )?;
        Ok(())
    }

    /// Keep working memory within size/time limits (`beam.py` `_trim_working_memory` L3499-L3524).
    /// Consolidated rows are exempt — the "originals stay" contract means the TTL window only
    /// bounds not-yet-consolidated content.
    fn trim_working_memory(&self, conn: &Connection) -> Result<()> {
        let cutoff = chrono::Utc::now()
            - chrono::Duration::seconds((self.config.working_memory_ttl_hours * 3600.0) as i64);
        conn.execute(
            "DELETE FROM working_memory \
             WHERE session_id = ?1 \
               AND consolidated_at IS NULL \
               AND ( \
                 timestamp < ?2 OR \
                 id NOT IN ( \
                     SELECT id FROM working_memory \
                     WHERE session_id = ?1 AND consolidated_at IS NULL \
                     ORDER BY timestamp DESC \
                     LIMIT ?3 \
                 ) \
               )",
            params![
                self.config.session_id,
                cutoff.to_rfc3339(),
                self.config.working_memory_max_items as i64,
            ],
        )?;
        Ok(())
    }

    /// Auto-generate temporal annotations for a memory (`beam.py` `_add_temporal_triple`
    /// L3471-L3494): the write date as an `occurred_on` annotation, plus the source kind as a
    /// `has_source` annotation for non-conversational sources. Best-effort.
    fn add_temporal_triple(
        &self,
        conn: &Connection,
        memory_id: &str,
        timestamp: &str,
        source: &str,
    ) {
        let date_str = &timestamp[..timestamp.len().min(10)];
        if let Err(e) = annotations::add(conn, memory_id, "occurred_on", date_str, "", 1.0) {
            tracing::debug!(error = %e, "occurred_on annotation failed (non-fatal)");
        }
        if !source.is_empty() && !matches!(source, "conversation" | "user" | "assistant") {
            if let Err(e) = annotations::add(conn, memory_id, "has_source", source, "", 1.0) {
                tracing::debug!(error = %e, "has_source annotation failed (non-fatal)");
            }
        }
    }

    /// Extract regex entities and store them as `mentions` annotations at confidence 0.8
    /// (`beam.py` `_extract_and_store_entities` L1309-L1339). Gated by `extract_entities`;
    /// best-effort.
    fn extract_and_store_entities(&self, conn: &Connection, memory_id: &str, content: &str) {
        let entity_list = entities::extract_entities_regex(content);
        if entity_list.is_empty() {
            return;
        }
        if let Err(e) =
            annotations::add_many(conn, memory_id, "mentions", &entity_list, "regex", 0.8)
        {
            tracing::debug!(error = %e, "entity mention annotations failed (non-fatal)");
        }
    }

    /// Extract gist + facts, store them in the graph, and consolidate veracity (`beam.py`
    /// `_ingest_graph_and_veracity` L3311-L3356). Non-blocking — knowledge failures never affect
    /// memory storage.
    pub(crate) fn ingest_graph_and_veracity(
        &self,
        conn: &Connection,
        memory_id: &str,
        content: &str,
        veracity_label: &str,
    ) {
        let result: Result<()> = (|| {
            let gist = episodic_graph::extract_gist(content, memory_id);
            episodic_graph::store_gist(conn, &gist, memory_id)?;
            let facts = episodic_graph::extract_facts(content, memory_id);
            for fact in &facts {
                episodic_graph::store_fact(conn, fact, memory_id, &self.config.session_id)?;
            }
            // Link gist -> fact `ctx` edges (`beam.py` L3329-L3337).
            for fact in &facts {
                episodic_graph::add_edge(
                    conn,
                    &episodic_graph::GraphEdge {
                        source: gist.id.clone(),
                        target: fact.id.clone(),
                        edge_type: "ctx".to_string(),
                        weight: fact.confidence,
                    },
                )?;
            }
            // Veracity-weighted consolidation reuses the extracted facts, passing the memory's
            // veracity verbatim (`beam.py` L3341-L3356).
            for fact in &facts {
                veracity::consolidate_fact(
                    conn,
                    &fact.subject,
                    &fact.predicate,
                    &fact.object,
                    veracity_label,
                    memory_id,
                )?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            tracing::debug!(error = %e, "graph/veracity ingestion failed (non-fatal)");
        }
        self.proactively_link(conn, memory_id, content);
    }

    /// Auto-create graph edges between a new memory and related existing memories (`beam.py`
    /// `_proactively_link` L3358-L3468). Two zero-LLM strategies — FTS content similarity
    /// (`related_to`, weight `1 - i*0.2`) and entity co-occurrence via shared `mentions`
    /// (`references`, weight 0.8) — gated by `proactive_linking` (`MNEMOSYNE_PROACTIVE_LINKING`).
    fn proactively_link(&self, conn: &Connection, memory_id: &str, content: &str) {
        if !self.config.proactive_linking {
            return;
        }
        // ── Strategy 1: content similarity via direct FTS5 ──
        let similarity: Result<()> = (|| {
            let keywords: Vec<String> = content
                .split_whitespace()
                .map(|w| {
                    w.to_lowercase()
                        .trim_matches(|c: char| ".,!?;:'\"()[]{}".contains(c))
                        .to_string()
                })
                .filter(|w| w.len() > 2 && !LINK_STOP_WORDS.contains(&w.as_str()))
                .filter(|w| w.chars().all(|c| c.is_alphabetic()))
                .collect();
            if keywords.len() < 3 {
                return Ok(());
            }
            let fts_query = keywords.join(" OR ");
            let similar_ids: Vec<String> = {
                let mut stmt = conn.prepare(
                    "SELECT DISTINCT id FROM fts_working WHERE fts_working MATCH ?1 \
                     ORDER BY rank LIMIT 5",
                )?;
                let ids: Vec<String> = stmt
                    .query_map(params![fts_query], |r| r.get::<_, String>(0))?
                    .filter_map(|r| r.ok())
                    .filter(|id| id != memory_id)
                    .collect();
                ids
            };
            for (i, rid) in similar_ids.iter().take(5).enumerate() {
                let existing: Option<i64> = conn
                    .query_row(
                        "SELECT 1 FROM graph_edges WHERE (source = ?1 AND target = ?2) \
                         OR (source = ?2 AND target = ?1)",
                        params![memory_id, rid],
                        |r| r.get(0),
                    )
                    .optional()?;
                if existing.is_some() {
                    continue;
                }
                let weight = (1.0 - i as f64 * 0.2).max(0.1);
                episodic_graph::add_edge(
                    conn,
                    &episodic_graph::GraphEdge {
                        source: memory_id.to_string(),
                        target: rid.clone(),
                        edge_type: "related_to".to_string(),
                        weight,
                    },
                )?;
            }
            Ok(())
        })();
        if let Err(e) = similarity {
            tracing::debug!(error = %e, "proactive similarity linking failed (non-fatal)");
        }

        // ── Strategy 2: entity overlap via shared `mentions` annotations ──
        let entity_overlap: Result<()> = (|| {
            let values: Vec<String> = {
                let mut stmt = conn.prepare(
                    "SELECT value FROM annotations WHERE memory_id = ?1 AND kind = 'mentions'",
                )?;
                let vals = stmt
                    .query_map(params![memory_id], |r| r.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                vals
            };
            for entity_val in values {
                let related: Vec<String> = {
                    let mut stmt = conn.prepare(
                        "SELECT DISTINCT memory_id FROM annotations \
                         WHERE kind = 'mentions' AND value = ?1 AND memory_id != ?2 LIMIT 5",
                    )?;
                    let ids = stmt
                        .query_map(params![entity_val, memory_id], |r| r.get::<_, String>(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    ids
                };
                for rid in related {
                    let existing: Option<i64> = conn
                        .query_row(
                            "SELECT 1 FROM graph_edges WHERE source = ?1 AND target = ?2 \
                             AND edge_type = 'references'",
                            params![memory_id, rid],
                            |r| r.get(0),
                        )
                        .optional()?;
                    if existing.is_some() {
                        continue;
                    }
                    episodic_graph::add_edge(
                        conn,
                        &episodic_graph::GraphEdge {
                            source: memory_id.to_string(),
                            target: rid,
                            edge_type: "references".to_string(),
                            weight: 0.8,
                        },
                    )?;
                }
            }
            Ok(())
        })();
        if let Err(e) = entity_overlap {
            tracing::debug!(error = %e, "proactive entity linking failed (non-fatal)");
        }
    }

    /// The stable per-bank device identity for the event log (`sync.py` L628-L640): explicit config
    /// wins later (sync feature); else load from `sync_meta`, else generate and persist
    /// `device-<8 hex>`.
    pub(crate) fn device_id(&self, conn: &Connection) -> String {
        self.device_id
            .get_or_init(|| {
                let stored: Option<String> = conn
                    .query_row(
                        "SELECT value FROM sync_meta WHERE key = 'device_id'",
                        [],
                        |r| r.get(0),
                    )
                    .optional()
                    .ok()
                    .flatten();
                if let Some(id) = stored {
                    return id;
                }
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let seed = format!("{nanos}:{}:{:p}", std::process::id(), self);
                let id = format!("device-{}", &util::memory_id(&seed)[..8]);
                let _ = conn.execute(
                    "INSERT OR REPLACE INTO sync_meta (key, value) VALUES ('device_id', ?1)",
                    params![id],
                );
                id
            })
            .clone()
    }

    /// Append a mutation event to the `memory_events` log (`beam.py` `_emit_event` L2813 +
    /// `sync.py` `SyncEngine.log_event` L703-L767 folded into the engine as the always-on
    /// append-only event log). `operation` is one of `CREATE`/`UPDATE`/`DELETE`/`CONSOLIDATE`.
    /// Fire-and-forget: event-log failures must never block memory operations.
    pub(crate) fn emit_event(
        &self,
        conn: &Connection,
        operation: &str,
        memory_id: &str,
        content: Option<&str>,
        source: &str,
        importance: f64,
    ) {
        let timestamp = util::now_iso();
        let payload = content.map(|c| {
            serde_json::json!({
                "content": c,
                "source": source,
                "importance": importance,
                "session_id": self.config.session_id,
            })
            .to_string()
        });
        if !SUPPRESS_EVENT_LOG.with(|c| c.get()) {
            let device_id = self.device_id(conn);
            let seq = self
                .event_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let event_id = util::memory_id(&format!(
                "{device_id}|{memory_id}|{operation}|{timestamp}|{seq}"
            ));
            // Deterministic dedup hash (`sync.py` `_compute_event_hash` L692-L699).
            let parent_ids = "[]";
            let preimage = format!(
                "{memory_id}|{operation}|{timestamp}|{device_id}|{}|{parent_ids}|{}",
                payload.as_deref().unwrap_or(""),
                py_float(importance),
            );
            let event_hash = {
                use sha2::{Digest, Sha256};
                let mut h = Sha256::new();
                h.update(preimage.as_bytes());
                format!("{:x}", h.finalize())
            };
            let res = conn.execute(
                "INSERT INTO memory_events \
                 (event_id, memory_id, operation, timestamp, device_id, payload, parent_event_ids, \
                  importance, expiry, event_hash) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)",
                params![
                    event_id, memory_id, operation, timestamp, device_id, payload, parent_ids,
                    importance, event_hash,
                ],
            );
            if let Err(e) = res {
                tracing::debug!(error = %e, operation, "memory_events insert failed (non-fatal)");
            }
        }
        // In-process stream fanout when a host enabled it (`beam.py` `_emit_event` L2813-L2835):
        // CREATE/UPDATE/CONSOLIDATE map to the three Python emission sites; DELETE has no
        // Python-side stream event. Failures are contained inside `MemoryStream::emit`.
        if let Some(stream) = self.stream_if_active() {
            use crate::streaming::EventType;
            let event_type = match operation {
                "CREATE" => Some(EventType::MemoryAdded),
                "UPDATE" => Some(EventType::MemoryUpdated),
                "CONSOLIDATE" => Some(EventType::MemoryConsolidated),
                _ => None,
            };
            if let Some(event_type) = event_type {
                stream.emit(crate::streaming::MemoryEvent {
                    event_type,
                    memory_id: memory_id.to_string(),
                    timestamp,
                    session_id: Some(self.config.session_id.clone()),
                    content: content.map(str::to_string),
                    source: Some(source.to_string()),
                    importance: Some(importance),
                    metadata: None,
                    delta: None,
                });
            }
        }
    }

    /// Merge an LLM extraction result into the knowledge layer for a stored memory (`extraction.py`
    /// host-LLM path L203-L264). LLM entities/triples are layered **on top of** the always-on regex
    /// baseline: entities become higher-confidence `mentions` annotations (source `llm`), SPO
    /// triples become `facts` + `consolidated_facts`, and free-text statements become `fact`
    /// annotations. Annotation/fact stores dedupe, so re-ingesting an item the regex pass already
    /// captured is a no-op upsert. Best-effort: a malformed item is skipped, never failing the turn.
    pub fn ingest_extracted(
        &self,
        memory_id: &str,
        extracted: &crate::extract::Extracted,
    ) -> Result<()> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();

        let entity_list: Vec<String> = extracted
            .entities
            .iter()
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty())
            .collect();
        if !entity_list.is_empty() {
            annotations::add_many(&conn, memory_id, "mentions", &entity_list, "llm", 0.9)?;
        }

        for (n, t) in extracted.triples.iter().enumerate() {
            let (subject, predicate, object) =
                (t.subject.trim(), t.predicate.trim(), t.object.trim());
            if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                continue;
            }
            let fact = episodic_graph::Fact {
                id: format!("fact_{memory_id}_llm_{n}"),
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                object: object.to_string(),
                timestamp: now.clone(),
                confidence: t.confidence,
            };
            episodic_graph::store_fact(&conn, &fact, memory_id, &self.config.session_id)?;
            veracity::consolidate_fact(&conn, subject, predicate, object, "inferred", memory_id)?;
        }

        // The shared write-side fact filter (`annotations.py` `filter_facts`, applied by
        // `_extract_and_store_facts` beam L1365 before `add_many(kind="fact")`).
        let statements = annotations::filter_facts(
            &extracted
                .facts
                .iter()
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>(),
        );
        if !statements.is_empty() {
            annotations::add_many(&conn, memory_id, "fact", &statements, "llm", 0.9)?;
        }
        Ok(())
    }
}
