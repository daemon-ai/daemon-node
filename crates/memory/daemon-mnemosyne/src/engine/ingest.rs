// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Write path for the BEAM [`Engine`]: `remember`/`remember_with_vector`, deterministic + LLM
//! knowledge ingestion, and proactive co-occurrence linking. Split out of `engine.rs` (W-MNEMO).

use super::*;
use crate::dynamics::typed_memory;
use crate::knowledge::{annotations, entities, episodic_graph, temporal, veracity};
use crate::{memoria, sanitize, util};
use rusqlite::{params, Connection};

impl Engine {
    /// Store a memory in the working tier (`beam.py` `remember` L2836), keyword-only (no vector).
    /// Equivalent to [`Engine::remember_with_vector`] with no embedding.
    pub fn remember(&self, content: &str, args: &RememberArgs) -> Result<String> {
        self.remember_with_vector(content, args, None, "")
    }

    /// Store a memory in the working tier, optionally persisting a precomputed embedding into
    /// `memory_embeddings` (the f32-BLOB-as-JSON fallback store).
    ///
    /// The embedding is computed by the caller (the async [`MnemosyneProvider`] hooks) and passed in,
    /// so the synchronous engine never blocks on a model call. Current slice: sanitize + classify +
    /// insert + vector write; dedup and expanded knowledge ingestion remain port-spec follow-ups.
    pub fn remember_with_vector(
        &self,
        content: &str,
        args: &RememberArgs,
        vector: Option<&[f32]>,
        model: &str,
    ) -> Result<String> {
        // Sanitize first (`beam.py` L2874-L2880): binary/oversized/high-entropy payloads spill to the
        // blob store, leaving a placeholder + `{"_blob": {...}}` metadata persisted on the row.
        let (content, blob_meta) = sanitize::sanitize_content(content, &self.config.blob_dir());
        let metadata_json = if blob_meta.as_object().map(|m| m.is_empty()).unwrap_or(true) {
            "{}".to_string()
        } else {
            serde_json::to_string(&serde_json::json!({ "_blob": blob_meta }))?
        };
        let id = util::memory_id(&format!("{}:{}", self.config.session_id, content));
        let memory_type = typed_memory::classify(&content).as_str();
        let now = util::now_iso();
        // Deterministic temporal extraction (`temporal_parser.py`): populate the event_date columns
        // so recall + degradation can reason over when an event occurred, not just when it was stored.
        let temporal = temporal::extract_temporal(&content);
        let temporal_tags_json = serde_json::to_string(&temporal.temporal_tags)?;
        // Multi-agent identity columns (`beam.py` ctor L2616-L2618 / write L2974): stamp the row with
        // the configured author and the channel (defaulting to the session, per the BEAM ctor).
        let channel_id = self
            .config
            .channel_id
            .clone()
            .unwrap_or_else(|| self.config.session_id.clone());
        let conn = self.store.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO working_memory \
             (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
              memory_type, scope, event_date, event_date_precision, temporal_tags, \
              author_id, author_type, channel_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                id,
                content,
                args.source,
                now,
                self.config.session_id,
                args.importance,
                metadata_json,
                args.veracity,
                memory_type,
                args.scope,
                temporal.event_date,
                temporal.event_date_precision,
                temporal_tags_json,
                self.config.author_id,
                self.config.author_type,
                channel_id,
            ],
        )?;
        if let Some(vector) = vector {
            let embedding_json = serde_json::to_string(vector)?;
            conn.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) \
                 VALUES (?1, ?2, ?3)",
                params![id, embedding_json, model],
            )?;
        }
        self.ingest_knowledge(
            &conn,
            &IngestItem {
                memory_id: &id,
                content: &content,
                veracity: &args.veracity,
            },
        )?;
        // Always-on MEMORIA regex extraction (`beam.py` L3027-L3033): populate the specialist
        // tables for structured retrieval. Best-effort — extraction must never block storage.
        if let Err(e) = memoria::extract_and_store(&conn, &self.config.session_id, &content, 0, &id)
        {
            tracing::debug!(error = %e, "memoria extraction failed (non-fatal)");
        }
        self.audit(&conn, "remember", Some(&id), None);
        // Invalidate the enhanced-recall query cache (`beam.py` L3041-L3043). Only touched when the
        // cache was actually opened (enhanced recall used), so Base mode pays nothing.
        if let Some(cache) = self.query_cache.get() {
            cache.invalidate();
        }
        Ok(id)
    }

    /// Deterministic knowledge ingestion for a freshly-stored memory (`beam.py` write-path
    /// extraction L3300+): regex entities become `mentions` annotations, regex SPO triples become
    /// `facts` + `consolidated_facts`, and shared-entity co-occurrence draws `references` edges to
    /// prior memories (bounded fan-out). All keyed by `memory_id`. No LLM extraction (P2).
    pub(crate) fn ingest_knowledge(&self, conn: &Connection, item: &IngestItem) -> Result<()> {
        let IngestItem {
            memory_id,
            content,
            veracity,
        } = *item;
        let entity_list = entities::extract_entities_regex(content);
        if !entity_list.is_empty() {
            annotations::add_many(conn, memory_id, "mentions", &entity_list, "regex", 1.0)?;
        }

        // Rule-based episode gist (participants/temporal/location/emotion) for the polyphonic graph
        // voice (`episodic_graph.py` `extract_gist` L165-L275).
        let gist = episodic_graph::extract_gist(content, memory_id);
        episodic_graph::store_gist(conn, &gist, memory_id)?;

        for fact in episodic_graph::extract_facts(content, memory_id) {
            episodic_graph::store_fact(conn, &fact, memory_id, &self.config.session_id)?;
            // Regex facts are inferred unless the memory itself was stated/tool/imported.
            let fact_veracity = if veracity == "unknown" {
                "inferred"
            } else {
                veracity
            };
            veracity::consolidate_fact(
                conn,
                &fact.subject,
                &fact.predicate,
                &fact.object,
                fact_veracity,
                memory_id,
            )?;
        }

        self.link_cooccurring(conn, memory_id, &entity_list)?;
        Ok(())
    }

    /// Proactive linking: connect `memory_id` to earlier memories that mention a shared entity
    /// (bounded fan-out per entity). Shared by the regex and LLM ingest paths.
    fn link_cooccurring(
        &self,
        conn: &Connection,
        memory_id: &str,
        entity_list: &[String],
    ) -> Result<()> {
        for entity in entity_list {
            let mentions = annotations::query_by_kind(conn, "mentions", Some(entity), false)?;
            let mut linked = 0usize;
            for other in mentions {
                if other.memory_id == memory_id {
                    continue;
                }
                episodic_graph::add_edge(
                    conn,
                    &episodic_graph::GraphEdge {
                        source: memory_id.to_string(),
                        target: other.memory_id,
                        edge_type: "references".to_string(),
                        weight: 0.8,
                    },
                )?;
                linked += 1;
                if linked >= MAX_COOCCURRENCE_EDGES_PER_ENTITY {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Merge an LLM extraction result into the knowledge layer for a stored memory (`extraction.py`
    /// host-LLM path L203-L264). LLM entities/triples are layered **on top of** the always-on regex
    /// baseline ([`Self::ingest_knowledge`]): entities become higher-confidence `mentions`
    /// annotations (source `llm`), SPO triples become `facts` + `consolidated_facts`, and free-text
    /// statements become `fact` annotations. Annotation/fact stores dedupe, so re-ingesting an item
    /// the regex pass already captured is a no-op upsert. Best-effort: a malformed item is skipped,
    /// never failing the turn.
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

        let statements: Vec<String> = extracted
            .facts
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() >= 5)
            .collect();
        if !statements.is_empty() {
            annotations::add_many(&conn, memory_id, "fact", &statements, "llm", 0.9)?;
        }

        self.link_cooccurring(&conn, memory_id, &entity_list)?;
        Ok(())
    }
}
