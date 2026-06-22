//! The BEAM engine facade — port of `beam.py` `BeamMemory` (the `remember`/`recall`/`get_context`/
//! `sleep` surface, L2836 / L5027 / L3526 / L7576) plus the `memory.py` facade.
//!
//! As-built: `remember`/`get_context` plus a hybrid `recall` that gathers candidates across the
//! **working and episodic** tiers from FTS5 (`fts_working`/`fts_episodes`, BM25), the stored
//! embeddings (cosine), and a recency/importance fallback scan, then scores them
//! ([`crate::recall::scoring`]) with the FTS-blended lexical relevance, vector similarity, the MIB
//! `binary_bonus`, and the tier/veracity multipliers — merged, content-deduped, and MMR-diversified.
//! [`Engine::consolidate`] is a minimal WM->episodic promotion (no LLM summarization/degradation).
//! Knowledge ingestion (graph/fact bonuses) and full `sleep` remain TODO (port-spec P1).

use crate::config::{MnemosyneConfig, RecallMode};
use crate::dynamics::{typed_memory, weibull};
use crate::error::Result;
use crate::knowledge::{annotations, entities, episodic_graph, temporal, veracity};
use crate::recall::query_cache::QueryCache;
use crate::recall::{mmr, polyphonic, query_intent, scoring, synonyms};
use crate::store::Store;
use crate::{binary_vectors, sanitize, util};
use rusqlite::{params, Connection};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Max co-occurrence edges drawn per shared entity at ingest, bounding the proactive-link fan-out
/// (`beam.py` `_proactively_link` is similarly capped).
const MAX_COOCCURRENCE_EDGES_PER_ENTITY: usize = 10;

/// The vector-similarity floor that lets a vector-only hit survive the lexical gate (mirrors the
/// episodic candidate-drop rule `lexical < floor && sim < 0.65 -> drop`, `beam.py` L5720+).
const VEC_SIM_FLOOR: f64 = 0.65;

/// Working-memory TTL in hours (`beam.py` `WORKING_MEMORY_TTL_HOURS` L269). Sleep's age cutoff is
/// half this (rows older than `TTL/2` are eligible for consolidation).
const WORKING_MEMORY_TTL_HOURS: i64 = 168;

/// Max working rows claimed per sleep pass (`beam.py` `SLEEP_BATCH_SIZE` L276).
const SLEEP_BATCH_SIZE: usize = 5000;

/// Tier 1->2 degradation age in days (`beam.py` `TIER2_DAYS` L281).
const TIER2_DAYS: i64 = 30;

/// Tier 2->3 degradation age in days (`beam.py` `TIER3_DAYS` L282).
const TIER3_DAYS: i64 = 180;

/// Max rows degraded per tier per pass (`beam.py` `DEGRADE_BATCH_SIZE` L286).
const DEGRADE_BATCH_SIZE: usize = 100;

/// Tier-3 compression target length (`beam.py` `TIER3_MAX_CHARS` L288).
const TIER3_MAX_CHARS: usize = 300;

/// A pending consolidation group (rows sharing a `source`), produced by [`Engine::sleep_plan`] and
/// summarized by the async provider before [`Engine::finish_sleep`] writes the episodic summary.
#[derive(Clone, Debug)]
pub struct SleepGroup {
    /// The shared ingestion source.
    pub source: String,
    /// The claimed working-memory row ids.
    pub ids: Vec<String>,
    /// The row contents, in timestamp order (the summarization input).
    pub contents: Vec<String>,
    /// Aggregated scope: `global` if any member is global, else `session` (`beam.py` L7686).
    pub scope: String,
    /// Aggregated veracity label (`beam.py` `aggregate_veracity` L7701).
    pub veracity: String,
    /// Aggregated `valid_until` (earliest member expiry, if any).
    pub valid_until: Option<String>,
}

/// Bank statistics (`beam.py` `stats`).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Stats {
    /// Live working-memory rows for the session/global scope.
    pub working: i64,
    /// Live episodic rows.
    pub episodic: i64,
    /// Episodic rows at tier 1/2/3.
    pub episodic_tier1: i64,
    /// Episodic rows at tier 2.
    pub episodic_tier2: i64,
    /// Episodic rows at tier 3.
    pub episodic_tier3: i64,
    /// Stored consolidated facts.
    pub facts: i64,
    /// Open temporal triples.
    pub triples: i64,
    /// Recorded conflicts.
    pub conflicts: i64,
}

/// A lightweight diagnostics summary (`beam.py` `health`/diagnostics).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Diagnostics {
    /// Working rows awaiting consolidation.
    pub pending_consolidation: i64,
    /// Episodic rows with a stored dense embedding.
    pub embedded_episodic: i64,
    /// Total episodic rows.
    pub episodic: i64,
    /// The most recent consolidation timestamp, if any.
    pub last_consolidation: Option<String>,
    /// Unresolved conflicts.
    pub open_conflicts: i64,
}

/// The outcome of a [`Engine::sleep`] pass.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct SleepReport {
    /// Working rows consolidated.
    pub items_consolidated: usize,
    /// Episodic summaries written.
    pub summaries_created: usize,
    /// Summaries that used an injected LLM (vs the AAAK fallback).
    pub llm_used: usize,
    /// Tier 1->2 degradations.
    pub tier1_to_tier2: usize,
    /// Tier 2->3 degradations.
    pub tier2_to_tier3: usize,
}

/// Which BEAM tier a row lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    /// Hot, recent, auto-injected context.
    Working,
    /// Long-term consolidated memory.
    Episodic,
}

/// A recalled / stored memory row (the `recall` result shape, `beam.py` L5996+).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemoryRow {
    /// Memory id.
    pub id: String,
    /// Content text.
    pub content: String,
    /// Ingestion source.
    pub source: String,
    /// ISO timestamp.
    pub timestamp: String,
    /// Importance `[0, 1]`.
    pub importance: f64,
    /// Trust label (`stated`/`inferred`/`tool`/`imported`/`unknown`).
    pub veracity: String,
    /// Trust tier (`STATED`/`DERIVED`/...).
    pub trust_tier: String,
    /// Which tier the row came from.
    pub tier: Tier,
    /// The episodic tier level (`1`/`2`/`3`); working rows are always `1` (`beam.py` L5931).
    pub tier_level: i64,
    /// The recall score (0 for direct fetches).
    pub score: f64,
}

/// An unresolved `(subject, predicate)` contradiction awaiting LLM validation in sleep.
#[derive(Clone, Debug)]
pub struct PendingConflict {
    /// The `conflicts` row id.
    pub conflict_id: i64,
    /// The newer fact's consolidated id (the one being inserted).
    pub newer_fact_id: String,
    /// The newer fact reconstructed as `subject predicate object`.
    pub newer_text: String,
    /// The older/existing fact's consolidated id.
    pub older_fact_id: String,
    /// The older fact reconstructed as `subject predicate object`.
    pub older_text: String,
}

/// Arguments for [`Engine::remember`] (`beam.py` `remember` L2836).
#[derive(Clone, Debug)]
pub struct RememberArgs {
    /// Ingestion source (default `conversation`).
    pub source: String,
    /// Importance `[0, 1]` (default 0.5).
    pub importance: f64,
    /// Scope: `session` (default) or `global`. Note: the column default is `global` but
    /// `remember()` defaults to `session` (`beam.py` L2838).
    pub scope: String,
    /// Trust label (default `unknown`).
    pub veracity: String,
}

impl Default for RememberArgs {
    fn default() -> Self {
        Self {
            source: "conversation".to_string(),
            importance: 0.5,
            scope: "session".to_string(),
            veracity: "unknown".to_string(),
        }
    }
}

/// The BEAM engine over a single bank store.
pub struct Engine {
    store: Store,
    config: MnemosyneConfig,
    /// Whether the bank is file-backed (enables the on-disk `query_cache.db`).
    persistent: bool,
    /// Lazily-opened 5-tier semantic query cache (enhanced recall only).
    query_cache: OnceLock<QueryCache>,
}

impl Engine {
    /// Open the engine for the configured bank.
    pub fn open(config: MnemosyneConfig) -> Result<Self> {
        let store = Store::open(config.bank_db_path())?;
        Ok(Self {
            store,
            config,
            persistent: true,
            query_cache: OnceLock::new(),
        })
    }

    /// Open an ephemeral in-memory engine (tests).
    pub fn open_in_memory(config: MnemosyneConfig) -> Result<Self> {
        let store = Store::open_in_memory()?;
        Ok(Self {
            store,
            config,
            persistent: false,
            query_cache: OnceLock::new(),
        })
    }

    /// The lazily-opened query cache (`query_cache.db` next to the bank when persistent, else
    /// memory-only). Used by [`Engine::recall_enhanced`].
    fn query_cache(&self) -> &QueryCache {
        self.query_cache.get_or_init(|| {
            if self.persistent {
                let cache_path = self
                    .config
                    .bank_db_path()
                    .parent()
                    .map(|p| p.join("query_cache.db"));
                QueryCache::open(cache_path.as_deref())
            } else {
                QueryCache::open(None)
            }
        })
    }

    /// The active session id.
    pub fn session_id(&self) -> &str {
        &self.config.session_id
    }

    /// Whether the opt-in tier-2 LLM conflict detector is enabled (`MNEMOSYNE_LLM_CONFLICT_DETECTION`).
    pub fn llm_conflict_detection(&self) -> bool {
        self.config.llm_conflict_detection
    }

    /// Unresolved `(subject, predicate)` contradictions recorded during consolidation, each with the
    /// reconstructed older/newer fact text for LLM validation (`fact_a` is the newer fact,
    /// `fact_b` the existing/older one — see [`veracity::record_conflict`] call in `consolidate_fact`).
    pub fn pending_conflicts(&self) -> Result<Vec<PendingConflict>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id, a.id, a.subject || ' ' || a.predicate || ' ' || a.object, \
                    b.id, b.subject || ' ' || b.predicate || ' ' || b.object \
             FROM conflicts c \
             JOIN consolidated_facts a ON a.id = c.fact_a_id \
             JOIN consolidated_facts b ON b.id = c.fact_b_id \
             WHERE c.resolution IS NULL",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PendingConflict {
                conflict_id: r.get::<_, i64>(0)?,
                newer_fact_id: r.get::<_, String>(1)?,
                newer_text: r.get::<_, String>(2)?,
                older_fact_id: r.get::<_, String>(3)?,
                older_text: r.get::<_, String>(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Resolve a recorded conflict (`veracity_consolidation.py` resolution path). On a confirmed
    /// conflict the older fact is marked `superseded_by` the newer one; either way the `conflicts`
    /// row is stamped with the LLM resolution + timestamp so it is not re-validated.
    pub fn resolve_conflict(
        &self,
        conflict_id: i64,
        confirmed: bool,
        winner_fact_id: &str,
        loser_fact_id: &str,
    ) -> Result<()> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        if confirmed {
            conn.execute(
                "UPDATE consolidated_facts SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
                params![winner_fact_id, now, loser_fact_id],
            )?;
        }
        let resolution = if confirmed { "llm_confirmed" } else { "llm_rejected" };
        conn.execute(
            "UPDATE conflicts SET resolution = ?1, resolved_at = ?2 WHERE id = ?3",
            params![resolution, now, conflict_id],
        )?;
        Ok(())
    }

    /// Store a memory in the working tier (`beam.py` `remember` L2836), keyword-only (no vector).
    /// Equivalent to [`Engine::remember_with_vector`] with no embedding.
    pub fn remember(&self, content: &str, args: &RememberArgs) -> Result<String> {
        self.remember_with_vector(content, args, None, "")
    }

    /// Store a memory in the working tier, optionally persisting a precomputed embedding into
    /// `memory_embeddings` (the f32-BLOB-as-JSON fallback store).
    ///
    /// The embedding is computed by the caller (the async [`MnemosyneProvider`] hooks) and passed in,
    /// so the synchronous engine never blocks on a model call. Scaffold: sanitize + classify +
    /// insert + vector write (dedup and knowledge ingestion remain TODO).
    pub fn remember_with_vector(
        &self,
        content: &str,
        args: &RememberArgs,
        vector: Option<&[f32]>,
        model: &str,
    ) -> Result<String> {
        let (content, _meta) = sanitize::sanitize_content(content);
        let id = util::memory_id(&format!("{}:{}", self.config.session_id, content));
        let memory_type = typed_memory::classify(&content).as_str();
        let now = util::now_iso();
        // Deterministic temporal extraction (`temporal_parser.py`): populate the event_date columns
        // so recall + degradation can reason over when an event occurred, not just when it was stored.
        let temporal = temporal::extract_temporal(&content);
        let temporal_tags_json = serde_json::to_string(&temporal.temporal_tags)?;
        let conn = self.store.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO working_memory \
             (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
              memory_type, scope, event_date, event_date_precision, temporal_tags) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                id,
                content,
                args.source,
                now,
                self.config.session_id,
                args.importance,
                args.veracity,
                memory_type,
                args.scope,
                temporal.event_date,
                temporal.event_date_precision,
                temporal_tags_json,
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
        self.ingest_knowledge(&conn, &id, &content, &args.veracity)?;
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
    fn ingest_knowledge(
        &self,
        conn: &Connection,
        memory_id: &str,
        content: &str,
        veracity: &str,
    ) -> Result<()> {
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
            let fact_veracity = if veracity == "unknown" { "inferred" } else { veracity };
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
    pub fn ingest_extracted(&self, memory_id: &str, extracted: &crate::extract::Extracted) -> Result<()> {
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

    /// Auto-inject context: global then session-local working memory ordered by importance/recency
    /// (`beam.py` `get_context` L3526-L3606).
    pub fn get_context(&self, limit: usize) -> Result<Vec<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(MemoryRow {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get(4)?,
                    veracity: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    trust_tier: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                    tier: Tier::Working,
                    tier_level: 1,
                    score: 0.0,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Linear-hybrid recall over working memory (`beam.py` `recall` L5027), keyword-only. Equivalent
    /// to [`Engine::recall_with_vector`] with no query vector.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<MemoryRow>> {
        self.recall_with_vector(query, top_k, None)
    }

    /// Hybrid lexical + FTS5 + vector recall across the working **and** episodic tiers (`beam.py`
    /// `recall` L5027, candidate gathering L2423-L2597 / finalize L5996-L6119).
    ///
    /// Each tier gathers candidates from three sources — an FTS5 `MATCH` (BM25-ranked), the stored
    /// embeddings (cosine), and a recency/importance fallback scan — then scores them: working rows
    /// with [`scoring::working_memory_score`] over an FTS-blended relevance ([`scoring::blend_fts`]),
    /// episodic rows with [`scoring::episodic_score`] (tier + veracity multipliers, MIB binary
    /// bonus). A row survives the candidate gate if it clears the lexical floor, is an FTS hit, **or**
    /// its vector similarity clears [`VEC_SIM_FLOOR`]. Tiers are merged, deduped by content, MMR-
    /// diversified for `>=4`-token queries, and the surviving rows have their recall stats bumped.
    /// With `query_vector = None` the vector source is skipped (pure lexical + FTS recall).
    pub fn recall_with_vector(
        &self,
        query: &str,
        top_k: usize,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        match self.config.recall_mode {
            RecallMode::Base => self.recall_base(query, top_k, query_vector, scoring::DEFAULT_WEIGHTS),
            RecallMode::Enhanced => self.recall_enhanced(query, top_k, query_vector),
            RecallMode::Polyphonic => self.recall_polyphonic(query, top_k, query_vector),
        }
    }

    /// The base hybrid cross-tier recall with explicit `(vec, fts, importance)` weights. This is the
    /// faithful port of `beam.py` `recall`; the enhanced/polyphonic pipelines build on it.
    fn recall_base(
        &self,
        query: &str,
        top_k: usize,
        query_vector: Option<&[f32]>,
        weights: (f64, f64, f64),
    ) -> Result<Vec<MemoryRow>> {
        let q_tokens = tokenize(query);
        let q_entities = entities::extract_entities_regex(query);
        let floor = scoring::lexical_floor(q_tokens.len());
        let conn = self.store.conn.lock().unwrap();

        let mut scored =
            self.gather_working(&conn, &q_tokens, &q_entities, top_k, floor, query_vector, weights)?;
        let episodic = self
            .gather_episodic(&conn, &q_tokens, &q_entities, top_k, floor, query_vector, weights)?;
        scored.extend(episodic);

        // Graph expansion: pull in memories that mention a query entity (or sit within two graph
        // hops of one) but were missed by the lexical/FTS/vector gates (`beam.py` L5760-L5793).
        let present: HashSet<String> = scored.iter().map(|r| r.id.clone()).collect();
        let injected = self.inject_entity_candidates(&conn, &q_entities, &present)?;
        scored.extend(injected);

        // Cross-tier dedup by normalized content, keeping the higher-scoring row (`beam.py` L6003).
        dedup_by_content(&mut scored);
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Diversity rerank for multi-token queries (`beam.py` L6061), else a plain top-k slice.
        let selected: Vec<MemoryRow> = if q_tokens.len() >= 4 && scored.len() > 1 {
            let items: Vec<(String, f64)> =
                scored.iter().map(|r| (r.content.clone(), r.score)).collect();
            mmr::mmr_rerank(&items, mmr::DEFAULT_LAMBDA, top_k)
                .into_iter()
                .map(|i| scored[i].clone())
                .collect()
        } else {
            scored.truncate(top_k);
            scored
        };

        self.bump_recall(&conn, &selected)?;
        Ok(selected)
    }

    /// Enhanced recall (`beam.py` `recall_enhanced` L6177-L6328): classify the query intent and bias
    /// the hybrid weights, synonym-expand the query, consult the 5-tier query cache, run base recall
    /// over the expanded query, Weibull-rescore by memory type (`score*0.7 + wb*0.3`), MMR-diversify,
    /// and cache the result. Associative graph expansion is off by default in Python, and base recall
    /// already injects entity/graph candidates, so it is not re-run here.
    fn recall_enhanced(
        &self,
        query: &str,
        top_k: usize,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        // 1. Intent classification -> weight bias.
        let intent = query_intent::classify_intent(query);
        let weights = if intent == query_intent::Intent::General {
            scoring::DEFAULT_WEIGHTS
        } else {
            query_intent::adjust_weights(scoring::DEFAULT_WEIGHTS, intent)
        };

        // 2. Synonym expansion (broadens FTS/lexical candidate generation).
        let expanded = synonyms::expand_query(query);

        // 3. Query cache check (keyed on the original query).
        if let Some(mut cached) = self.query_cache().get(query, query_vector) {
            cached.truncate(top_k);
            return Ok(cached);
        }

        // 4. Base recall over the expanded query, gathering a wider pool.
        let mut results = self.recall_base(&expanded, top_k * 2, query_vector, weights)?;

        // 5. Weibull re-scoring by memory type.
        self.weibull_rescore(&mut results)?;

        // 6-7. Sort, then MMR-diversify down to `top_k`.
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let final_results: Vec<MemoryRow> = if results.len() > 1 {
            let items: Vec<(String, f64)> =
                results.iter().map(|r| (r.content.clone(), r.score)).collect();
            mmr::mmr_rerank(&items, mmr::DEFAULT_LAMBDA, top_k)
                .into_iter()
                .map(|i| results[i].clone())
                .collect()
        } else {
            results.truncate(top_k);
            results
        };

        // 8. Cache the result for next time.
        self.query_cache().put(query, &final_results, query_vector);
        Ok(final_results)
    }

    /// Blend the per-type Weibull temporal boost into each row's score (`beam.py` L6266-L6278):
    /// `score = score*0.7 + weibull_boost*0.3`. The memory type is read from the row's tier table;
    /// missing/`unknown` types fall back to `general`.
    fn weibull_rescore(&self, rows: &mut [MemoryRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let conn = self.store.conn.lock().unwrap();
        for row in rows.iter_mut() {
            let table = match row.tier {
                Tier::Working => "working_memory",
                Tier::Episodic => "episodic_memory",
            };
            let mt: Option<String> = conn
                .query_row(
                    &format!("SELECT memory_type FROM {table} WHERE id = ?1"),
                    params![row.id],
                    |r| r.get(0),
                )
                .ok();
            let mut memory_type = mt.unwrap_or_default();
            if memory_type.is_empty() || memory_type == "unknown" {
                memory_type = "general".to_string();
            }
            let wb = weibull::weibull_boost(age_hours(&row.timestamp), &memory_type);
            row.score = row.score * 0.7 + wb * 0.3;
        }
        Ok(())
    }

    /// Four-voice polyphonic recall (`polyphonic_recall.py`, `MNEMOSYNE_POLYPHONIC_RECALL=1`):
    /// gathers a **vector** voice (cosine normalized `(cos+1)/2`, top 20), a **graph** voice (query
    /// entities seed `facts` subjects at `0.6`, then `ctx`-edge traversal at `0.4/depth`), a **fact**
    /// voice (consolidated `facts` whose subject is a query word, `confidence >= 0.5`), and a
    /// **temporal** voice (last-7-day working rows, `exp(-age_days/7)*importance`, only on temporal
    /// keywords), then fuses them with RRF ([`polyphonic::fuse`]), diversity-reranks, and resolves
    /// the surviving ids to rows. Voice weights stay metadata-only (fusion is pure RRF).
    fn recall_polyphonic(
        &self,
        query: &str,
        top_k: usize,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        use polyphonic::VoiceHit;
        let conn = self.store.conn.lock().unwrap();

        // Voice 1: vector (cosine over stored embeddings, normalized to [0, 1], top 20).
        let mut vector_hits: Vec<VoiceHit> = Vec::new();
        if let Some(q) = query_vector {
            let stored = load_embeddings(&conn)?;
            let mut sims: Vec<(String, f64)> = stored
                .iter()
                .map(|(id, v)| (id.clone(), (daemon_core::cosine(q, v) as f64 + 1.0) / 2.0))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(20);
            vector_hits = sims.into_iter().map(|(id, s)| VoiceHit { memory_id: id, score: s }).collect();
        }

        // Voice 2: graph (entity-seeded gists @0.6 + fact subjects @conf*0.5, then ctx-edge
        // traversal @0.4/depth from all seeds).
        let q_entities = entities::extract_entities_regex(query);
        let mut graph_hits: Vec<VoiceHit> = Vec::new();
        let mut seen_graph: HashSet<String> = HashSet::new();
        let mut seed_ids: HashSet<String> = HashSet::new();
        for ent in &q_entities {
            for (mid, _text) in episodic_graph::find_gists_by_participant(&conn, ent)? {
                if seen_graph.insert(mid.clone()) {
                    graph_hits.push(VoiceHit { memory_id: mid.clone(), score: 0.6 });
                }
                seed_ids.insert(mid);
            }
            for (mid, conf) in self.facts_for_subject(&conn, ent, 0.0)? {
                if seen_graph.insert(mid.clone()) {
                    graph_hits.push(VoiceHit { memory_id: mid.clone(), score: conf * 0.5 });
                }
                seed_ids.insert(mid);
            }
        }
        for seed in &seed_ids {
            for rel in episodic_graph::find_related_memories(&conn, seed, 2, "ctx", 0.3)? {
                if !seed_ids.contains(&rel.memory_id) && seen_graph.insert(rel.memory_id.clone()) {
                    graph_hits.push(VoiceHit {
                        memory_id: rel.memory_id,
                        score: 0.4 / rel.depth as f64,
                    });
                }
            }
        }

        // Voice 3: fact (query words matched against consolidated `facts` subjects).
        let mut fact_hits: Vec<VoiceHit> = Vec::new();
        let mut seen_fact: HashSet<String> = HashSet::new();
        for word in tokenize(query) {
            if word.chars().count() < 3 {
                continue;
            }
            for (mid, conf) in self.facts_for_subject(&conn, &word, 0.5)? {
                if seen_fact.insert(mid.clone()) {
                    fact_hits.push(VoiceHit { memory_id: mid, score: conf });
                }
            }
        }

        // Voice 4: temporal (recent working rows, only when the query has a temporal cue).
        let mut temporal_hits: Vec<VoiceHit> = Vec::new();
        if has_temporal_keyword(query) {
            let week_ago = (chrono::Utc::now() - chrono::Duration::days(7)).to_rfc3339();
            let mut stmt = conn.prepare(
                "SELECT id, timestamp, importance FROM working_memory \
                 WHERE timestamp > ?1 AND superseded_by IS NULL \
                   AND (session_id = ?2 OR scope = 'global') \
                 ORDER BY timestamp DESC LIMIT 20",
            )?;
            let rows = stmt.query_map(params![week_ago, self.config.session_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, f64>(2)?))
            })?;
            for row in rows.flatten() {
                let (id, ts, imp) = row;
                let age_days = age_hours(&ts).unwrap_or(0.0) / 24.0;
                let tscore = (-age_days / 7.0).exp() * imp;
                temporal_hits.push(VoiceHit { memory_id: id, score: tscore });
            }
        }

        let fused = polyphonic::fuse(&[
            ("vector", vector_hits),
            ("graph", graph_hits),
            ("fact", fact_hits),
            ("temporal", temporal_hits),
        ]);
        let diversified = polyphonic::diversity_rerank(fused, top_k);

        let mut out: Vec<MemoryRow> = Vec::new();
        for f in diversified {
            let row = match self.fetch_working(&conn, &f.memory_id)? {
                Some(r) => Some(r),
                None => self.fetch_episodic(&conn, &f.memory_id)?,
            };
            if let Some(mut r) = row {
                r.score = f.combined_score;
                out.push(r);
            }
        }
        self.bump_recall(&conn, &out)?;
        Ok(out)
    }

    /// `(memory_id, confidence)` for `facts` whose subject matches `subject` at or above
    /// `min_confidence` (the polyphonic fact voice).
    fn facts_for_subject(
        &self,
        conn: &Connection,
        subject: &str,
        min_confidence: f64,
    ) -> Result<Vec<(String, f64)>> {
        let mut stmt = conn.prepare(
            "SELECT source_msg_id, confidence FROM facts \
             WHERE subject = ?1 COLLATE NOCASE AND confidence >= ?2 AND source_msg_id IS NOT NULL",
        )?;
        let rows = stmt.query_map(params![subject, min_confidence], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
        })?;
        Ok(rows.flatten().collect())
    }

    /// Gather + score working-memory candidates (FTS5 ∪ vector ∪ recency fallback), with the
    /// knowledge-layer graph/fact bonuses and entity/fact multipliers (`beam.py` L5760-L5793).
    #[allow(clippy::too_many_arguments)]
    fn gather_working(
        &self,
        conn: &Connection,
        q_tokens: &[String],
        q_entities: &[String],
        top_k: usize,
        floor: f64,
        query_vector: Option<&[f32]>,
        weights: (f64, f64, f64),
    ) -> Result<Vec<MemoryRow>> {
        // Base candidates: the recency/importance fallback scan (limit 2000, `beam.py` L5262), plus
        // any FTS5 hits that fall outside that window.
        let mut rows = self.scan_working(conn, 2000)?;
        let mut seen: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let fts = self.fts_search(
            conn,
            "SELECT id, bm25(fts_working) FROM fts_working \
             WHERE fts_working MATCH ?1 ORDER BY bm25(fts_working) LIMIT ?2",
            q_tokens,
            (top_k * 3).max(50),
        )?;
        for id in fts.keys() {
            if seen.insert(id.clone()) {
                if let Some(row) = self.fetch_working(conn, id)? {
                    rows.push(row);
                }
            }
        }

        let stored = match query_vector {
            Some(_) => load_embeddings(conn)?,
            None => HashMap::new(),
        };
        let (_vw, _fw, iw) = weights;

        let mut scored = Vec::new();
        for mut row in rows {
            let lexical = lexical_relevance(q_tokens, &row.content);
            let nfts = fts.get(&row.id).copied().unwrap_or(0.0);
            let vec_sim = match (query_vector, stored.get(&row.id)) {
                (Some(q), Some(v)) => daemon_core::cosine(q, v) as f64,
                _ => 0.0,
            };
            if lexical < floor && vec_sim < VEC_SIM_FLOOR && nfts <= 0.0 {
                continue;
            }
            let relevance = scoring::blend_fts(lexical, nfts, floor);
            let decay = scoring::recency_decay(age_hours(&row.timestamp));
            let bonuses = self.knowledge_bonuses(conn, &row.id, q_entities)?;
            let mut base = scoring::working_memory_score(relevance, row.importance, iw, vec_sim, decay);
            base += bonuses.graph_bonus + bonuses.fact_bonus;
            base = bonuses.apply_multipliers(base);
            row.score = base * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        Ok(scored)
    }

    /// Gather + score episodic candidates (FTS5 ∪ vector ∪ recency fallback), with the MIB binary
    /// bonus and the tier/veracity post-multipliers (`beam.py` L5720-L5976).
    #[allow(clippy::too_many_arguments)]
    fn gather_episodic(
        &self,
        conn: &Connection,
        q_tokens: &[String],
        q_entities: &[String],
        top_k: usize,
        floor: f64,
        query_vector: Option<&[f32]>,
        weights: (f64, f64, f64),
    ) -> Result<Vec<MemoryRow>> {
        let mut rows = self.scan_episodic(conn, 2000)?;
        let mut seen: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let fts = self.fts_search(
            conn,
            "SELECT e.id, bm25(fts_episodes) FROM fts_episodes f \
             JOIN episodic_memory e ON e.rowid = f.rowid \
             WHERE fts_episodes MATCH ?1 ORDER BY bm25(fts_episodes) LIMIT ?2",
            q_tokens,
            (top_k * 3).max(20),
        )?;
        for id in fts.keys() {
            if seen.insert(id.clone()) {
                if let Some(row) = self.fetch_episodic(conn, id)? {
                    rows.push(row);
                }
            }
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let stored = match query_vector {
            Some(_) => load_embeddings(conn)?,
            None => HashMap::new(),
        };
        let binaries = self.load_binary_vectors(conn)?;
        let q_bin = query_vector.map(binary_vectors::maximally_informative_binarization);

        let mut scored = Vec::new();
        for mut row in rows {
            let lexical = lexical_relevance(q_tokens, &row.content);
            let nfts = fts.get(&row.id).copied().unwrap_or(0.0);
            let sim = match (query_vector, stored.get(&row.id)) {
                (Some(q), Some(v)) => daemon_core::cosine(q, v) as f64,
                _ => 0.0,
            };
            // Weak-signal gate (`beam.py` L5720): drop unless lexical, FTS, or vector say keep.
            if lexical < floor && sim < VEC_SIM_FLOOR && nfts <= 0.0 {
                continue;
            }
            let binary_bonus = match (&q_bin, binaries.get(&row.id)) {
                (Some(qb), Some(rb)) => {
                    let dist = binary_vectors::hamming_distance(qb, rb);
                    binary_vectors::binary_bonus(dist as f64 / binary_vectors::EMBEDDING_DIM as f64)
                }
                _ => 0.0,
            };
            let decay = scoring::recency_decay(age_hours(&row.timestamp));
            let bonuses = self.knowledge_bonuses(conn, &row.id, q_entities)?;
            let base = scoring::episodic_score(
                sim,
                nfts,
                row.importance,
                lexical,
                decay,
                weights,
                bonuses.graph_bonus,
                bonuses.fact_bonus,
                binary_bonus,
            );
            let base = bonuses.apply_multipliers(base);
            row.score = base
                * scoring::tier_weight(row.tier_level)
                * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        Ok(scored)
    }

    /// Promote unconsolidated working-memory rows into the episodic tier (a minimal slice of
    /// `beam.py` `sleep`/consolidation L7576: no LLM summarization or tier degradation yet). Each
    /// promoted row is copied into `episodic_memory` at tier 1 — computing its MIB `binary_vector`
    /// from any stored embedding — its source working row is marked `consolidated_at`, and a
    /// `consolidation_log` entry is written. Returns the number of rows promoted.
    pub fn consolidate(&self) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        let embeddings = load_embeddings(&conn)?;
        let mut pending: Vec<EpisodicSeed> = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT id, content, source, timestamp, importance, veracity, trust_tier, scope, \
                        memory_type, event_date, event_date_precision, temporal_tags \
                 FROM working_memory \
                 WHERE consolidated_at IS NULL AND session_id = ?1 AND superseded_by IS NULL",
            )?;
            let rows = stmt.query_map(params![self.config.session_id], |r| {
                Ok(EpisodicSeed {
                    wm_id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get(4)?,
                    veracity: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    trust_tier: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                    scope: r.get::<_, Option<String>>(7)?.unwrap_or_else(|| "global".into()),
                    memory_type: r
                        .get::<_, Option<String>>(8)?
                        .unwrap_or_else(|| "unknown".into()),
                    event_date: r.get::<_, Option<String>>(9)?,
                    event_date_precision: r
                        .get::<_, Option<String>>(10)?
                        .unwrap_or_else(|| "unknown".into()),
                    temporal_tags: r.get::<_, Option<String>>(11)?.unwrap_or_else(|| "[]".into()),
                })
            })?;
            for row in rows {
                pending.push(row?);
            }
        }
        if pending.is_empty() {
            return Ok(0);
        }

        let now = util::now_iso();
        let mut count = 0usize;
        for seed in &pending {
            let ep_id =
                util::memory_id(&format!("episodic:{}:{}", self.config.session_id, seed.content));
            let binary = embeddings
                .get(&seed.wm_id)
                .map(|v| binary_vectors::maximally_informative_binarization(v));
            conn.execute(
                "INSERT OR IGNORE INTO episodic_memory \
                 (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
                  memory_type, tier, binary_vector, scope, trust_tier, summary_of, \
                  event_date, event_date_precision, temporal_tags) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, ?8, 1, ?9, ?10, ?11, '', ?12, ?13, ?14)",
                params![
                    ep_id,
                    seed.content,
                    seed.source,
                    seed.timestamp,
                    self.config.session_id,
                    seed.importance,
                    seed.veracity,
                    seed.memory_type,
                    binary,
                    seed.scope,
                    seed.trust_tier,
                    seed.event_date,
                    seed.event_date_precision,
                    seed.temporal_tags,
                ],
            )?;
            conn.execute(
                "UPDATE working_memory SET consolidated_at = ?2 WHERE id = ?1",
                params![seed.wm_id, now],
            )?;
            // Mirror the deterministic knowledge layer onto the episodic id so the episodic recall
            // tier carries its own entity/fact/graph signals.
            self.ingest_knowledge(&conn, &ep_id, &seed.content, &seed.veracity)?;
            count += 1;
        }
        let preview: String = pending
            .iter()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
            .chars()
            .take(200)
            .collect();
        conn.execute(
            "INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview) \
             VALUES (?1, ?2, ?3)",
            params![self.config.session_id, count as i64, preview],
        )?;
        Ok(count)
    }

    /// The summarization prompt for one source group (`beam.py` sleep LLM path L7749). Built here so
    /// the async provider and the engine agree on the contract.
    pub fn summary_prompt(contents: &[String]) -> String {
        format!(
            "Summarize the following related memories into a single concise note that preserves all \
             durable facts, names, decisions, and dates. Be terse; no preamble.\n\n{}\n\nSummary:",
            contents.join("\n- ")
        )
    }

    /// Plan a sleep pass (`beam.py` sleep L7597-L7676): select eligible working rows (older than the
    /// `TTL/2` cutoff unless `force`, skipping pinned/consolidated rows, oldest-first, capped at
    /// [`SLEEP_BATCH_SIZE`]), **atomically claim** them (set `consolidated_at`/`consolidation_claimed_at`
    /// gated on `consolidated_at IS NULL` for crash-/concurrency-safety), and group the claimed rows
    /// by source. The caller (the async provider) summarizes each group and hands the summaries to
    /// [`Engine::finish_sleep`]. Returns an empty vec when nothing is eligible.
    pub fn sleep_plan(&self, force: bool) -> Result<Vec<SleepGroup>> {
        let conn = self.store.conn.lock().unwrap();
        let cutoff = if force {
            "9999-12-31T23:59:59+00:00".to_string()
        } else {
            (chrono::Utc::now() - chrono::Duration::hours(WORKING_MEMORY_TTL_HOURS / 2))
                .to_rfc3339()
        };

        struct Claimable {
            id: String,
            content: String,
            source: String,
            scope: String,
            valid_until: Option<String>,
            veracity: String,
        }
        let candidates: Vec<Claimable> = {
            let mut stmt = conn.prepare(
                "SELECT id, content, COALESCE(source, 'conversation'), \
                        COALESCE(scope, 'session'), valid_until, COALESCE(veracity, 'unknown') \
                 FROM working_memory \
                 WHERE COALESCE(session_id, 'default') = ?1 \
                   AND timestamp < ?2 \
                   AND consolidated_at IS NULL \
                   AND (pinned IS NULL OR pinned = 0) \
                   AND superseded_by IS NULL \
                 ORDER BY timestamp ASC LIMIT ?3",
            )?;
            let rows = stmt.query_map(
                params![self.config.session_id, cutoff, SLEEP_BATCH_SIZE as i64],
                |r| {
                    Ok(Claimable {
                        id: r.get(0)?,
                        content: r.get(1)?,
                        source: r.get(2)?,
                        scope: r.get(3)?,
                        valid_until: r.get(4)?,
                        veracity: r.get(5)?,
                    })
                },
            )?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Atomic claim: mark consolidated_at gated on it still being NULL, then keep only the rows we
        // actually won (a concurrent sleep may have claimed some).
        let now = util::now_iso();
        let mut claimed: Vec<Claimable> = Vec::new();
        for c in candidates {
            let n = conn.execute(
                "UPDATE working_memory SET consolidated_at = ?2, consolidation_claimed_at = ?2 \
                 WHERE id = ?1 AND consolidated_at IS NULL",
                params![c.id, now],
            )?;
            if n == 1 {
                claimed.push(c);
            }
        }
        if claimed.is_empty() {
            return Ok(Vec::new());
        }

        // Group by source, aggregating scope/veracity/valid_until (`beam.py` L7674-L7703).
        let mut order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, SleepGroup> = HashMap::new();
        let mut veracities: HashMap<String, Vec<String>> = HashMap::new();
        for c in claimed {
            let g = groups.entry(c.source.clone()).or_insert_with(|| {
                order.push(c.source.clone());
                SleepGroup {
                    source: c.source.clone(),
                    ids: Vec::new(),
                    contents: Vec::new(),
                    scope: "session".to_string(),
                    veracity: "unknown".to_string(),
                    valid_until: None,
                }
            });
            g.ids.push(c.id);
            g.contents.push(c.content);
            if c.scope == "global" {
                g.scope = "global".to_string();
            }
            if let Some(vu) = c.valid_until {
                g.valid_until = Some(match g.valid_until.take() {
                    Some(existing) if existing < vu => existing,
                    _ => vu,
                });
            }
            veracities.entry(c.source).or_default().push(c.veracity);
        }
        for (source, vs) in &veracities {
            if let Some(g) = groups.get_mut(source) {
                g.veracity = veracity::aggregate_veracity(vs);
            }
        }
        Ok(order.into_iter().filter_map(|s| groups.remove(&s)).collect())
    }

    /// Write the episodic summaries for the claimed [`SleepGroup`]s (`beam.py` sleep L7784-L7824),
    /// then run tiered degradation. `summaries` maps a group's `source` to an LLM summary; a group
    /// with no entry falls back to the deterministic AAAK summary `[source] <aaak>`. Clears each
    /// group's `consolidation_claimed_at`, writes a `consolidation_log` row, and returns the report.
    pub fn finish_sleep(
        &self,
        groups: &[SleepGroup],
        summaries: &HashMap<String, String>,
    ) -> Result<SleepReport> {
        let mut report = SleepReport::default();
        if !groups.is_empty() {
            let conn = self.store.conn.lock().unwrap();
            for group in groups {
                let (summary, llm) = match summaries.get(&group.source) {
                    Some(s) if !s.trim().is_empty() => (s.trim().to_string(), true),
                    _ => (
                        format!("[{}] {}", group.source, crate::aaak::summarize_group(&group.contents)),
                        false,
                    ),
                };
                let ep_id = util::memory_id(&format!(
                    "episodic:{}:{}",
                    self.config.session_id, summary
                ));
                let summary_of = serde_json::to_string(&group.ids)?;
                conn.execute(
                    "INSERT OR IGNORE INTO episodic_memory \
                     (id, content, source, timestamp, session_id, importance, metadata_json, \
                      veracity, memory_type, tier, scope, summary_of, valid_until) \
                     VALUES (?1, ?2, 'sleep_consolidation', ?3, ?4, 0.6, '{}', ?5, 'unknown', 1, \
                             ?6, ?7, ?8)",
                    params![
                        ep_id,
                        summary,
                        util::now_iso(),
                        self.config.session_id,
                        group.veracity,
                        group.scope,
                        summary_of,
                        group.valid_until,
                    ],
                )?;
                self.ingest_knowledge(&conn, &ep_id, &summary, &group.veracity)?;
                let placeholders = group
                    .ids
                    .iter()
                    .map(|_| "?")
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "UPDATE working_memory SET consolidation_claimed_at = NULL WHERE id IN ({placeholders})"
                );
                let id_params: Vec<&dyn rusqlite::ToSql> =
                    group.ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
                conn.execute(&sql, id_params.as_slice())?;

                report.items_consolidated += group.ids.len();
                report.summaries_created += 1;
                if llm {
                    report.llm_used += 1;
                }
            }
            let method = if report.llm_used == report.summaries_created {
                "llm"
            } else if report.llm_used > 0 {
                "llm+aaak"
            } else {
                "aaak"
            };
            conn.execute(
                "INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview) \
                 VALUES (?1, ?2, ?3)",
                params![
                    self.config.session_id,
                    report.items_consolidated as i64,
                    format!(
                        "{} summaries ({method}) from {} items",
                        report.summaries_created, report.items_consolidated
                    ),
                ],
            )?;
        }

        let (t1, t2) = self.degrade_episodic()?;
        report.tier1_to_tier2 = t1;
        report.tier2_to_tier3 = t2;
        Ok(report)
    }

    /// Run a full sleep pass with the deterministic AAAK summary (no LLM). Equivalent to
    /// [`Engine::sleep_plan`] + [`Engine::finish_sleep`] with no LLM summaries. The async provider
    /// uses the split form to inject LLM summaries; this is the standalone/no-LLM entrypoint.
    pub fn sleep(&self, force: bool) -> Result<SleepReport> {
        let groups = self.sleep_plan(force)?;
        self.finish_sleep(&groups, &HashMap::new())
    }

    /// Tiered episodic degradation (`beam.py` `degrade_episodic` L7241-L7366): tier 1 rows older than
    /// [`TIER2_DAYS`] are AAAK-compressed and promoted to tier 2; tier 2 rows older than
    /// [`TIER3_DAYS`] are signal-compressed to <=[`TIER3_MAX_CHARS`] and promoted to tier 3. When a
    /// row's content changes its stale dense embedding is invalidated (dropped + binary vector
    /// cleared) so recall doesn't score against text that no longer exists. Recall already applies a
    /// tier weight, so degraded rows score down automatically. Returns `(tier1->2, tier2->3)` counts.
    pub fn degrade_episodic(&self) -> Result<(usize, usize)> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();

        // Tier 1 -> 2: AAAK-compress.
        let tier1: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, content FROM episodic_memory \
                 WHERE tier = 1 AND created_at < datetime('now', ?1) \
                 ORDER BY created_at ASC LIMIT ?2",
            )?;
            let rows = stmt.query_map(
                params![format!("-{TIER2_DAYS} days"), DEGRADE_BATCH_SIZE as i64],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut t1 = 0usize;
        for (id, content) in &tier1 {
            let compressed = crate::aaak::encode(content);
            let final_content: String = compressed.chars().take(800).collect();
            conn.execute(
                "UPDATE episodic_memory SET content = ?2, tier = 2, degraded_at = ?3 WHERE id = ?1",
                params![id, final_content, now],
            )?;
            if &final_content != content {
                self.invalidate_episodic_embedding(&conn, id)?;
            }
            t1 += 1;
        }

        // Tier 2 -> 3: signal-compress to TIER3_MAX_CHARS.
        let tier2: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, content FROM episodic_memory \
                 WHERE tier = 2 AND created_at < datetime('now', ?1) \
                 ORDER BY created_at ASC LIMIT ?2",
            )?;
            let rows = stmt.query_map(
                params![format!("-{TIER3_DAYS} days"), (DEGRADE_BATCH_SIZE / 2) as i64],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut t2 = 0usize;
        for (id, content) in &tier2 {
            let compressed = compress_to(content, TIER3_MAX_CHARS);
            conn.execute(
                "UPDATE episodic_memory SET content = ?2, tier = 3, degraded_at = ?3 WHERE id = ?1",
                params![id, compressed, now],
            )?;
            if &compressed != content {
                self.invalidate_episodic_embedding(&conn, id)?;
            }
            t2 += 1;
        }
        Ok((t1, t2))
    }

    /// Drop a degraded row's stale dense embedding + binary vector (`beam.py`
    /// `_refresh_episodic_embedding` invalidation path C18.b) so recall falls back to lexical/FTS
    /// until a fresh embedding is computed.
    fn invalidate_episodic_embedding(&self, conn: &Connection, id: &str) -> Result<()> {
        conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?1", params![id])?;
        conn.execute(
            "UPDATE episodic_memory SET binary_vector = NULL WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    // ── Tool-surface backing methods (`beam.py` get/update/forget/invalidate/validate/stats/...) ──

    /// Fetch a single live memory by id, working tier first then episodic (`beam.py` `get`).
    pub fn get(&self, id: &str) -> Result<Option<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        if let Some(row) = self.fetch_working(&conn, id)? {
            return Ok(Some(row));
        }
        self.fetch_episodic(&conn, id)
    }

    /// Update a memory's `content` and/or `importance` in whichever tier holds it (`beam.py`
    /// `update`). FTS stays in sync via the content-update triggers. Returns whether a row changed.
    pub fn update(&self, id: &str, content: Option<&str>, importance: Option<f64>) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let mut changed = false;
        for table in ["working_memory", "episodic_memory"] {
            if let Some(c) = content {
                changed |= conn.execute(
                    &format!("UPDATE {table} SET content = ?2 WHERE id = ?1"),
                    params![id, c],
                )? > 0;
            }
            if let Some(imp) = importance {
                changed |= conn.execute(
                    &format!("UPDATE {table} SET importance = ?2 WHERE id = ?1"),
                    params![id, imp],
                )? > 0;
            }
        }
        Ok(changed)
    }

    /// Hard-delete a memory from both tiers plus its stored embedding (`beam.py` `forget`). FTS rows
    /// are removed by the delete triggers. Returns whether anything was deleted.
    pub fn forget(&self, id: &str) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let mut deleted = conn.execute("DELETE FROM working_memory WHERE id = ?1", params![id])?;
        deleted += conn.execute("DELETE FROM episodic_memory WHERE id = ?1", params![id])?;
        conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?1", params![id])?;
        Ok(deleted > 0)
    }

    /// Soft-invalidate a memory: stamp `valid_until` now and point `superseded_by` at an optional
    /// replacement (`beam.py` `invalidate` L7725). The row drops out of recall (which filters
    /// `valid_until IS NULL AND superseded_by IS NULL`). Returns whether a row changed.
    pub fn invalidate(&self, id: &str, replacement_id: Option<&str>) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        let mut changed = false;
        for table in ["working_memory", "episodic_memory"] {
            changed |= conn.execute(
                &format!(
                    "UPDATE {table} SET valid_until = ?2, superseded_by = ?3 \
                     WHERE id = ?1 AND valid_until IS NULL"
                ),
                params![id, now, replacement_id],
            )? > 0;
        }
        Ok(changed)
    }

    /// Record a human/agent validation action on a memory (`beam.py` `validate`). Appends a
    /// `memory_validations` row and bumps the row's `validation_count`/`validated_at`/`validator`.
    /// `action = "correct"` with `new_content` rewrites the content; `action = "reject"` invalidates
    /// the row. Returns whether the target memory exists.
    pub fn validate(
        &self,
        id: &str,
        action: &str,
        validator: Option<&str>,
        new_content: Option<&str>,
        note: Option<&str>,
    ) -> Result<bool> {
        let now = util::now_iso();
        {
            let conn = self.store.conn.lock().unwrap();
            let exists: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM working_memory WHERE id = ?1 \
                 UNION ALL SELECT 1 FROM episodic_memory WHERE id = ?1)",
                params![id],
                |r| r.get(0),
            )?;
            if !exists {
                return Ok(false);
            }
            conn.execute(
                "INSERT INTO memory_validations (memory_id, validator, validated_at, action, \
                 new_content, note) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, validator, now, action, new_content, note],
            )?;
            for table in ["working_memory", "episodic_memory"] {
                conn.execute(
                    &format!(
                        "UPDATE {table} SET validation_count = validation_count + 1, \
                         validated_at = ?2, validator = ?3 WHERE id = ?1"
                    ),
                    params![id, now, validator],
                )?;
            }
        }
        match action {
            "correct" => {
                if let Some(c) = new_content {
                    self.update(id, Some(c), None)?;
                }
            }
            "reject" => {
                self.invalidate(id, None)?;
            }
            _ => {}
        }
        Ok(true)
    }

    /// Bank statistics (`beam.py` `stats`): tier counts + structured-store sizes.
    pub fn stats(&self) -> Result<Stats> {
        let conn = self.store.conn.lock().unwrap();
        let count = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |r| r.get(0))?) };
        Ok(Stats {
            working: count(
                "SELECT COUNT(*) FROM working_memory WHERE valid_until IS NULL AND superseded_by IS NULL",
            )?,
            episodic: count(
                "SELECT COUNT(*) FROM episodic_memory WHERE valid_until IS NULL AND superseded_by IS NULL",
            )?,
            episodic_tier1: count("SELECT COUNT(*) FROM episodic_memory WHERE tier = 1")?,
            episodic_tier2: count("SELECT COUNT(*) FROM episodic_memory WHERE tier = 2")?,
            episodic_tier3: count("SELECT COUNT(*) FROM episodic_memory WHERE tier = 3")?,
            facts: count("SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NULL")?,
            triples: count("SELECT COUNT(*) FROM triples WHERE valid_until IS NULL")?,
            conflicts: count("SELECT COUNT(*) FROM conflicts")?,
        })
    }

    /// A lightweight diagnostics summary (`beam.py` `health`).
    pub fn diagnose(&self) -> Result<Diagnostics> {
        let conn = self.store.conn.lock().unwrap();
        Ok(Diagnostics {
            pending_consolidation: conn.query_row(
                "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NULL \
                 AND session_id = ?1 AND superseded_by IS NULL",
                params![self.config.session_id],
                |r| r.get(0),
            )?,
            embedded_episodic: conn.query_row(
                "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL",
                [],
                |r| r.get(0),
            )?,
            episodic: conn.query_row("SELECT COUNT(*) FROM episodic_memory", [], |r| r.get(0))?,
            last_consolidation: conn
                .query_row(
                    "SELECT MAX(created_at) FROM consolidation_log WHERE items_consolidated > 0",
                    [],
                    |r| r.get::<_, Option<String>>(0),
                )
                .unwrap_or(None),
            open_conflicts: conn.query_row(
                "SELECT COUNT(*) FROM conflicts WHERE resolution IS NULL",
                [],
                |r| r.get(0),
            )?,
        })
    }

    /// Write a scratchpad note for the session (`beam.py` scratchpad). Returns the row id.
    pub fn scratchpad_write(&self, content: &str) -> Result<String> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        let id = util::memory_id(&format!("scratch:{}:{}:{}", self.config.session_id, now, content));
        conn.execute(
            "INSERT OR REPLACE INTO scratchpad (id, content, session_id) VALUES (?1, ?2, ?3)",
            params![id, content, self.config.session_id],
        )?;
        Ok(id)
    }

    /// Read the session's scratchpad notes, newest first (`(id, content)` pairs).
    pub fn scratchpad_read(&self) -> Result<Vec<(String, String)>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content FROM scratchpad WHERE session_id = ?1 ORDER BY created_at DESC, id DESC",
        )?;
        let rows = stmt.query_map(params![self.config.session_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Clear the session's scratchpad. Returns the number of notes removed.
    pub fn scratchpad_clear(&self) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        Ok(conn.execute(
            "DELETE FROM scratchpad WHERE session_id = ?1",
            params![self.config.session_id],
        )?)
    }

    /// Export the session's working + episodic rows as a portable JSON bundle (`beam.py`
    /// `export`/sync surface). Knowledge structures are re-derivable from content on import.
    pub fn export(&self) -> Result<serde_json::Value> {
        let conn = self.store.conn.lock().unwrap();
        let dump = |table: &str| -> Result<Vec<serde_json::Value>> {
            let mut stmt = conn.prepare(&format!(
                "SELECT id, content, source, timestamp, importance, veracity, scope \
                 FROM {table} WHERE session_id = ?1 AND valid_until IS NULL AND superseded_by IS NULL"
            ))?;
            let rows = stmt.query_map(params![self.config.session_id], |r| {
                Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "content": r.get::<_, String>(1)?,
                    "source": r.get::<_, Option<String>>(2)?,
                    "timestamp": r.get::<_, Option<String>>(3)?,
                    "importance": r.get::<_, f64>(4)?,
                    "veracity": r.get::<_, Option<String>>(5)?,
                    "scope": r.get::<_, Option<String>>(6)?,
                }))
            })?;
            Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
        };
        Ok(json!({
            "version": 1,
            "session_id": self.config.session_id,
            "working_memory": dump("working_memory")?,
            "episodic_memory": dump("episodic_memory")?,
        }))
    }

    /// Import rows from an [`Engine::export`] bundle into this session, re-running knowledge + temporal
    /// ingestion for working rows. Returns the number of working rows imported.
    pub fn import(&self, bundle: &serde_json::Value) -> Result<usize> {
        let mut imported = 0usize;
        if let Some(rows) = bundle.get("working_memory").and_then(|v| v.as_array()) {
            for row in rows {
                let content = row.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if content.is_empty() {
                    continue;
                }
                let importance = row.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5);
                let scope = row
                    .get("scope")
                    .and_then(|v| v.as_str())
                    .unwrap_or("session")
                    .to_string();
                let veracity = row
                    .get("veracity")
                    .and_then(|v| v.as_str())
                    .unwrap_or("imported")
                    .to_string();
                self.remember_with_vector(
                    content,
                    &RememberArgs {
                        source: "import".to_string(),
                        importance,
                        scope,
                        veracity,
                    },
                    None,
                    "",
                )?;
                imported += 1;
            }
        }
        Ok(imported)
    }

    /// Graph neighbours of a memory within `depth` hops (`beam.py` `graph_query` /
    /// `episodic_graph::find_related_memories`).
    pub fn graph_query(&self, memory_id: &str, depth: usize) -> Result<Vec<episodic_graph::Related>> {
        let conn = self.store.conn.lock().unwrap();
        episodic_graph::find_related_memories(&conn, memory_id, depth.max(1), "", 0.0)
    }

    /// Add a manual graph edge between two memories (`beam.py` `graph_link` /
    /// `episodic_graph::add_edge`).
    pub fn graph_link(&self, source: &str, target: &str, edge_type: &str, weight: f64) -> Result<()> {
        let conn = self.store.conn.lock().unwrap();
        episodic_graph::add_edge(
            &conn,
            &episodic_graph::GraphEdge {
                source: source.to_string(),
                target: target.to_string(),
                edge_type: if edge_type.is_empty() {
                    "related_to".to_string()
                } else {
                    edge_type.to_string()
                },
                weight,
            },
        )
    }

    /// Add a temporal triple (`triples::add`).
    #[allow(clippy::too_many_arguments)]
    pub fn triple_add(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        valid_from: Option<&str>,
        valid_until: Option<&str>,
        source: &str,
        confidence: f64,
        supersede: bool,
    ) -> Result<i64> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::triples::add(
            &conn, subject, predicate, object, valid_from, valid_until, source, confidence, supersede,
        )
    }

    /// Expire open triples (`triples::end`).
    pub fn triple_end(
        &self,
        subject: &str,
        predicate: &str,
        object: Option<&str>,
        valid_until: Option<&str>,
    ) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::triples::end(&conn, subject, predicate, object, valid_until)
    }

    /// Query temporal triples valid at `as_of` (`triples::query`).
    pub fn triple_query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
        as_of: Option<&str>,
    ) -> Result<Vec<crate::knowledge::triples::Triple>> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::triples::query(&conn, subject, predicate, object, as_of)
    }

    /// Upsert a canonical identity fact (`canonical::remember`).
    pub fn canonical_remember(
        &self,
        owner_id: &str,
        category: &str,
        name: &str,
        body: &str,
        source: &str,
        confidence: f64,
    ) -> Result<(crate::knowledge::canonical::CanonicalRow, crate::knowledge::canonical::Status)> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::canonical::remember(&conn, owner_id, category, name, body, source, confidence)
    }

    /// Read live canonical facts for an owner (`canonical::current`).
    pub fn canonical_recall(
        &self,
        owner_id: &str,
        category: Option<&str>,
        name: Option<&str>,
    ) -> Result<Vec<crate::knowledge::canonical::CanonicalRow>> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::canonical::current(&conn, owner_id, category, name)
    }

    /// Retire a canonical fact slot (`canonical::forget`).
    pub fn canonical_forget(&self, owner_id: &str, category: &str, name: &str) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::canonical::forget(&conn, owner_id, category, name)
    }

    /// Run an FTS5 `MATCH` query (`sql` selecting `(id, bm25)`), returning `id -> normalized BM25`
    /// for the hits. An empty token list (or a query with no usable terms) yields no hits.
    fn fts_search(
        &self,
        conn: &Connection,
        sql: &str,
        q_tokens: &[String],
        limit: usize,
    ) -> Result<HashMap<String, f64>> {
        let Some(match_str) = fts_match_string(q_tokens) else {
            return Ok(HashMap::new());
        };
        let mut stmt = conn.prepare(sql)?;
        let mut map = HashMap::new();
        let rows = stmt.query_map(params![match_str, limit as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
        })?;
        for row in rows {
            let (id, bm25) = row?;
            map.insert(id, normalize_bm25(bm25));
        }
        Ok(map)
    }

    /// Recency/importance fallback scan over working memory (the candidate floor, scope-filtered).
    fn scan_working(&self, conn: &Connection, limit: usize) -> Result<Vec<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(working_row(r))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetch a single working row by id (for FTS hits beyond the fallback window), scope-filtered.
    fn fetch_working(&self, conn: &Connection, id: &str) -> Result<Option<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE id = ?1 AND (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?2 OR scope = 'global')",
        )?;
        let mut rows = stmt.query_map(params![id, self.config.session_id], |r| Ok(working_row(r)))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Recency/importance fallback scan over episodic memory, scope-filtered.
    fn scan_episodic(&self, conn: &Connection, limit: usize) -> Result<Vec<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier, tier \
             FROM episodic_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(episodic_row(r))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetch a single episodic row by id (for FTS hits beyond the fallback window), scope-filtered.
    fn fetch_episodic(&self, conn: &Connection, id: &str) -> Result<Option<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier, tier \
             FROM episodic_memory \
             WHERE id = ?1 AND (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?2 OR scope = 'global')",
        )?;
        let mut rows =
            stmt.query_map(params![id, self.config.session_id], |r| Ok(episodic_row(r)))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Load the packed MIB `binary_vector` blobs for episodic rows, keyed by memory id.
    fn load_binary_vectors(&self, conn: &Connection) -> Result<HashMap<String, Vec<u8>>> {
        let mut stmt = conn
            .prepare("SELECT id, binary_vector FROM episodic_memory WHERE binary_vector IS NOT NULL")?;
        let mut map = HashMap::new();
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        for row in rows {
            let (id, blob) = row?;
            map.insert(id, blob);
        }
        Ok(map)
    }

    /// Bump `recall_count` / `last_recalled` for the returned rows in their source tier (`beam.py`
    /// L6084-L6119).
    fn bump_recall(&self, conn: &Connection, rows: &[MemoryRow]) -> Result<()> {
        let now = util::now_iso();
        for row in rows {
            let table = match row.tier {
                Tier::Working => "working_memory",
                Tier::Episodic => "episodic_memory",
            };
            conn.execute(
                &format!(
                    "UPDATE {table} SET recall_count = recall_count + 1, last_recalled = ?2 \
                     WHERE id = ?1"
                ),
                params![row.id, now],
            )?;
        }
        Ok(())
    }

    /// Compute the knowledge-layer recall signals for a candidate keyed by `row_id`: the additive
    /// `graph_bonus` (incident `graph_edges`) and `fact_bonus` (query entities appearing in the
    /// row's `facts`), plus the entity (`*1.3`, capped) and fact (`*1.2`) post-multiplier flags
    /// (`beam.py` L5779-L5793). With no query entities all signals are inert.
    fn knowledge_bonuses(
        &self,
        conn: &Connection,
        row_id: &str,
        q_entities: &[String],
    ) -> Result<KnowledgeBonuses> {
        let edges = episodic_graph::edge_count(conn, row_id)?;
        let graph_bonus = scoring::graph_bonus(edges);
        if q_entities.is_empty() {
            return Ok(KnowledgeBonuses {
                graph_bonus,
                fact_bonus: 0.0,
                entity_match: false,
                fact_match: false,
            });
        }

        let mentions = annotations::query_by_memory(conn, row_id, Some("mentions"))?;
        let entity_match = q_entities.iter().any(|e| {
            mentions
                .iter()
                .any(|m| m.value.eq_ignore_ascii_case(e.as_str()))
        });

        let mut stmt =
            conn.prepare("SELECT subject, object FROM facts WHERE source_msg_id = ?1")?;
        let fact_terms: Vec<(String, String)> = stmt
            .query_map(params![row_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let fact_match_count = q_entities
            .iter()
            .filter(|e| {
                fact_terms.iter().any(|(s, o)| {
                    s.eq_ignore_ascii_case(e.as_str()) || o.eq_ignore_ascii_case(e.as_str())
                })
            })
            .count();

        Ok(KnowledgeBonuses {
            graph_bonus,
            fact_bonus: scoring::fact_bonus(fact_match_count),
            entity_match,
            fact_match: fact_match_count > 0,
        })
    }

    /// Inject working candidates that mention a query entity (or sit within two graph hops of one)
    /// but were missed by the lexical/FTS/vector gates (`beam.py` L5760-L5793). New candidates are
    /// scored with the entity-recall floor `(0.6 + 0.2*imp) * (0.7 + 0.3*decay) * veracity`.
    fn inject_entity_candidates(
        &self,
        conn: &Connection,
        q_entities: &[String],
        present: &HashSet<String>,
    ) -> Result<Vec<MemoryRow>> {
        if q_entities.is_empty() {
            return Ok(Vec::new());
        }
        let mut seeds: HashSet<String> = HashSet::new();
        for entity in q_entities {
            for ann in annotations::query_by_kind(conn, "mentions", Some(entity), false)? {
                seeds.insert(ann.memory_id);
            }
        }

        // Fuzzy entity matching (`entities.py` `find_similar_entities`, threshold 0.8): also seed
        // memories that mention a *similar* entity ("Acme" vs "Acme Corp", typos), not just exact
        // string equality. The known-entity universe is the deduped `mentions` annotation values.
        let all_mentions = annotations::query_by_kind(conn, "mentions", None, true)?;
        if !all_mentions.is_empty() {
            let mut value_to_memories: HashMap<String, Vec<String>> = HashMap::new();
            for ann in &all_mentions {
                value_to_memories.entry(ann.value.clone()).or_default().push(ann.memory_id.clone());
            }
            let known: Vec<String> = value_to_memories.keys().cloned().collect();
            for entity in q_entities {
                for (matched, _score) in entities::find_similar_entities(entity, &known, 0.8) {
                    if matched.eq_ignore_ascii_case(entity) {
                        continue; // exact matches already seeded above
                    }
                    if let Some(ids) = value_to_memories.get(&matched) {
                        seeds.extend(ids.iter().cloned());
                    }
                }
            }
        }

        // One graph expansion (depth 2) from the directly-mentioning seeds.
        let mut expanded: HashSet<String> = HashSet::new();
        for seed in &seeds {
            for rel in episodic_graph::find_related_memories(conn, seed, 2, "", 0.0)? {
                expanded.insert(rel.memory_id);
            }
        }
        seeds.extend(expanded);

        let mut out = Vec::new();
        for id in seeds {
            if present.contains(&id) {
                continue;
            }
            if let Some(mut row) = self.fetch_working(conn, &id)? {
                let decay = scoring::recency_decay(age_hours(&row.timestamp));
                let base = (0.6 + 0.2 * row.importance) * (0.7 + 0.3 * decay);
                row.score = base * scoring::veracity_multiplier(&row.veracity);
                out.push(row);
            }
        }
        Ok(out)
    }
}

/// The knowledge-layer recall signals for a single candidate.
struct KnowledgeBonuses {
    graph_bonus: f64,
    fact_bonus: f64,
    entity_match: bool,
    fact_match: bool,
}

impl KnowledgeBonuses {
    /// Apply the entity (`*1.3`, capped at 1.0) and fact (`*1.2`) multipliers to a base score
    /// (`beam.py` L5785-L5793).
    fn apply_multipliers(&self, base: f64) -> f64 {
        let mut s = base;
        if self.entity_match {
            s = (s * 1.3).min(1.0);
        }
        if self.fact_match {
            s *= 1.2;
        }
        s
    }
}

/// A working-memory row queued for promotion into the episodic tier ([`Engine::consolidate`]).
struct EpisodicSeed {
    wm_id: String,
    content: String,
    source: String,
    timestamp: String,
    importance: f64,
    veracity: String,
    trust_tier: String,
    scope: String,
    memory_type: String,
    event_date: Option<String>,
    event_date_precision: String,
    temporal_tags: String,
}

/// Compress `content` to at most `max_chars` characters for tier-3 degradation (`beam.py`
/// `_extract_key_signal`/truncation L7344-L7349): keep the head and mark elision.
fn compress_to(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let head: String = content.chars().take(max_chars).collect();
    format!("{head} [...]")
}

/// Load the stored f32 embeddings (`memory_embeddings.embedding_json`), keyed by memory id.
fn load_embeddings(conn: &Connection) -> Result<HashMap<String, Vec<f32>>> {
    let mut stmt = conn.prepare("SELECT memory_id, embedding_json FROM memory_embeddings")?;
    let mut map = HashMap::new();
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (id, json) = row?;
        if let Ok(vec) = serde_json::from_str::<Vec<f32>>(&json) {
            map.insert(id, vec);
        }
    }
    Ok(map)
}

/// Map a working-memory result row (`id, content, source, timestamp, importance, veracity,
/// trust_tier`) into a [`MemoryRow`] at tier [`Tier::Working`].
fn working_row(r: &rusqlite::Row<'_>) -> MemoryRow {
    MemoryRow {
        id: r.get(0).unwrap_or_default(),
        content: r.get(1).unwrap_or_default(),
        source: r.get::<_, Option<String>>(2).ok().flatten().unwrap_or_default(),
        timestamp: r.get::<_, Option<String>>(3).ok().flatten().unwrap_or_default(),
        importance: r.get(4).unwrap_or(0.5),
        veracity: r.get::<_, Option<String>>(5).ok().flatten().unwrap_or_default(),
        trust_tier: r.get::<_, Option<String>>(6).ok().flatten().unwrap_or_default(),
        tier: Tier::Working,
        tier_level: 1,
        score: 0.0,
    }
}

/// Map an episodic result row (working columns + `tier`) into a [`MemoryRow`] at tier
/// [`Tier::Episodic`], carrying the integer tier level for the post-multiplier.
fn episodic_row(r: &rusqlite::Row<'_>) -> MemoryRow {
    MemoryRow {
        id: r.get(0).unwrap_or_default(),
        content: r.get(1).unwrap_or_default(),
        source: r.get::<_, Option<String>>(2).ok().flatten().unwrap_or_default(),
        timestamp: r.get::<_, Option<String>>(3).ok().flatten().unwrap_or_default(),
        importance: r.get(4).unwrap_or(0.5),
        veracity: r.get::<_, Option<String>>(5).ok().flatten().unwrap_or_default(),
        trust_tier: r.get::<_, Option<String>>(6).ok().flatten().unwrap_or_default(),
        tier: Tier::Episodic,
        tier_level: r.get::<_, Option<i64>>(7).ok().flatten().unwrap_or(1),
        score: 0.0,
    }
}

/// Tokenize text into lowercase alphanumeric terms (the shared query/content tokenizer).
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Build an FTS5 `MATCH` expression from query tokens (`"a" OR "b" OR ...`), quoting each term so
/// punctuation/operators can never break the query. `None` when there are no usable terms.
fn fts_match_string(q_tokens: &[String]) -> Option<String> {
    if q_tokens.is_empty() {
        return None;
    }
    let parts: Vec<String> = q_tokens
        .iter()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    Some(parts.join(" OR "))
}

/// Map SQLite FTS5 `bm25()` (more-negative = better) onto `[0, 1)` (`raw / (1 + raw)`), so a missed
/// row contributes `0` and a strong match approaches `1`.
fn normalize_bm25(bm25: f64) -> f64 {
    let raw = (-bm25).max(0.0);
    raw / (1.0 + raw)
}

/// Normalized content key for cross-tier dedup (lowercased alphanumeric tokens, space-joined).
fn normalized_content(content: &str) -> String {
    tokenize(content).join(" ")
}

/// Drop duplicate rows sharing the same [`normalized_content`], keeping the highest-scoring one
/// (`beam.py` cross-tier summary dedup L6003).
fn dedup_by_content(rows: &mut Vec<MemoryRow>) {
    let mut best: HashMap<String, usize> = HashMap::new();
    let mut keep = vec![true; rows.len()];
    for i in 0..rows.len() {
        let key = normalized_content(&rows[i].content);
        match best.get(&key).copied() {
            Some(j) => {
                if rows[i].score > rows[j].score {
                    keep[j] = false;
                    best.insert(key, i);
                } else {
                    keep[i] = false;
                }
            }
            None => {
                best.insert(key, i);
            }
        }
    }
    let mut idx = 0;
    rows.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Lexical relevance (`beam.py` `_lexical_relevance` L1573-L1638): `(exact_token_hits + partial +
/// full_match)/len`, where a query token absent from the content earns `+0.75` for a
/// [`synonyms::recall_synonyms`] hit (beam's conservative `_RECALL_SYNONYMS` map, L1608-L1611), else
/// `+0.4` for a `>=4`-char substring overlap; a whole-query substring adds `1.0`. Clamped to `[0, 1]`.
fn lexical_relevance(query_tokens: &[String], content: &str) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let lc = content.to_lowercase();
    let content_tokens: HashSet<String> = tokenize(content).into_iter().collect();
    let mut num = 0.0;
    for t in query_tokens {
        if content_tokens.contains(t) {
            num += 1.0;
            continue;
        }
        let syns = synonyms::recall_synonyms(t);
        if !syns.is_empty() && syns.iter().any(|s| content_tokens.contains(*s)) {
            num += 0.75;
            continue;
        }
        if t.len() >= 4 && lc.contains(t.as_str()) {
            num += 0.4;
        }
    }
    if lc.contains(&query_tokens.join(" ")) {
        num += 1.0;
    }
    (num / query_tokens.len() as f64).min(1.0)
}

/// Temporal cue words that activate the polyphonic temporal voice (`polyphonic_recall.py`
/// `_temporal_voice` L628-L633).
fn has_temporal_keyword(query: &str) -> bool {
    const TEMPORAL_KEYWORDS: &[&str] = &[
        "yesterday", "today", "recent", "last", "latest", "this week", "this month", "ago",
        "before",
    ];
    let lower = query.to_lowercase();
    TEMPORAL_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Hours since an ISO timestamp (`None` if unparseable -> decay falls back to 0.5).
fn age_hours(timestamp: &str) -> Option<f64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(timestamp).ok()?;
    let now = chrono::Utc::now();
    let delta = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
    Some(delta.num_seconds().max(0) as f64 / 3600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        Engine::open_in_memory(MnemosyneConfig::default()).expect("engine")
    }

    #[test]
    fn remember_then_recall() {
        let e = engine();
        e.remember(
            "the authentication flow uses JWT tokens",
            &RememberArgs::default(),
        )
        .unwrap();
        e.remember("lunch was pizza", &RememberArgs::default())
            .unwrap();
        let hits = e.recall("authentication flow", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("authentication"));
    }

    #[test]
    fn session_scoping_over_shared_bank() {
        // Two engines over the *same* agent-wide bank, each bound to its own session id (the
        // per-session construction the composition layer's `MnemosyneBanks` performs). Session-scoped
        // rows must not leak across sessions, while `scope='global'` rows are visible to both.
        let dir = std::env::temp_dir().join(format!("mnemosyne-scope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = |sid: &str| MnemosyneConfig {
            data_dir: dir.clone(),
            session_id: sid.to_string(),
            ..MnemosyneConfig::default()
        };
        let s1 = Engine::open(cfg("s1")).expect("open s1");
        let s2 = Engine::open(cfg("s2")).expect("open s2");

        s1.remember("alpha private to one", &RememberArgs::default())
            .unwrap();
        s2.remember("beta private to two", &RememberArgs::default())
            .unwrap();
        s1.remember(
            "gamma shared globally",
            &RememberArgs {
                scope: "global".to_string(),
                ..Default::default()
            },
        )
        .unwrap();

        // Each session sees its own session-scoped row...
        assert!(!s1.recall("alpha", 5).unwrap().is_empty());
        assert!(!s2.recall("beta", 5).unwrap().is_empty());
        // ...but not the other session's.
        assert!(
            s1.recall("beta", 5).unwrap().is_empty(),
            "s1 must not see s2's row"
        );
        assert!(
            s2.recall("alpha", 5).unwrap().is_empty(),
            "s2 must not see s1's row"
        );
        // The global row is visible to both.
        assert!(!s1.recall("gamma", 5).unwrap().is_empty());
        assert!(
            !s2.recall("gamma", 5).unwrap().is_empty(),
            "global row visible across sessions"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn vector_recall_surfaces_semantic_match_lexical_misses() {
        let e = engine();
        // A query vector, one near-parallel memory vector (cos ~0.96) and one orthogonal — with
        // content that shares NO tokens with the query, so lexical recall finds nothing.
        let q = [1.0f32, 0.0, 0.0];
        let near = [0.96f32, 0.28, 0.0];
        let far = [0.0f32, 0.0, 1.0];
        e.remember_with_vector("alpha apple", &RememberArgs::default(), Some(&near), "mock")
            .unwrap();
        e.remember_with_vector("beta banana", &RememberArgs::default(), Some(&far), "mock")
            .unwrap();

        // Lexical-only recall for a disjoint query returns nothing.
        assert!(e.recall("zzz", 5).unwrap().is_empty());

        // Vector recall surfaces the semantically-close memory and ranks it first.
        let hits = e.recall_with_vector("zzz", 5, Some(&q)).unwrap();
        assert!(!hits.is_empty(), "vector recall should surface a match");
        assert_eq!(hits[0].content, "alpha apple");
        assert!(
            hits.iter().all(|h| h.content != "beta banana"),
            "orthogonal memory must not pass the vector gate"
        );
    }

    #[test]
    fn get_context_orders_by_importance() {
        let e = engine();
        e.remember(
            "low",
            &RememberArgs {
                importance: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        e.remember(
            "high",
            &RememberArgs {
                importance: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = e.get_context(10).unwrap();
        assert_eq!(ctx[0].content, "high");
    }

    #[test]
    fn lexical_relevance_scores() {
        let q = vec!["auth".to_string(), "flow".to_string()];
        // Both tokens present as whole words + full-query substring -> clamped to 1.0.
        assert!((lexical_relevance(&q, "the auth flow uses jwt") - 1.0).abs() < 1e-9);
        // One exact token of two -> 0.5.
        assert!((lexical_relevance(&q, "the auth subsystem") - 0.5).abs() < 1e-9);
        // A >=4-char substring (no whole-word match, and the full query is not a substring)
        // contributes the 0.4 partial: one of two tokens at 0.4 -> 0.2.
        let q2 = vec!["serialize".to_string(), "absent".to_string()];
        assert!((lexical_relevance(&q2, "the deserializer ran") - 0.2).abs() < 1e-9);
        // Disjoint query -> 0.0; empty query -> 0.0.
        assert_eq!(lexical_relevance(&q, "completely unrelated"), 0.0);
        assert_eq!(lexical_relevance(&[], "anything"), 0.0);
    }

    #[test]
    fn fts_surfaces_row_beyond_recency_window() {
        // Fill the recency/importance window (limit 2000) with high-importance filler that does NOT
        // contain the marker, then add one low-importance row that does. The marker row ranks 2001st
        // by importance, so it is *outside* the fallback scan — only the FTS5 candidate path can
        // surface it. (Under the old full-scan recall this row was unreachable.)
        let e = engine();
        for i in 0..2000 {
            e.remember(
                &format!("filler row number {i}"),
                &RememberArgs {
                    importance: 0.9,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        e.remember(
            "a unique zqxj marker lives here",
            &RememberArgs {
                importance: 0.1,
                ..Default::default()
            },
        )
        .unwrap();

        let hits = e.recall("zqxj", 5).unwrap();
        assert!(
            hits.iter().any(|h| h.content.contains("zqxj")),
            "FTS5 must surface a row outside the recency window"
        );
    }

    #[test]
    fn consolidation_populates_episodic_and_is_idempotent() {
        let e = engine();
        e.remember("blue-green deployment rollout strategy", &RememberArgs::default())
            .unwrap();
        e.remember("margherita pizza for lunch", &RememberArgs::default())
            .unwrap();

        assert_eq!(e.consolidate().unwrap(), 2, "both WM rows promoted");
        assert_eq!(e.consolidate().unwrap(), 0, "already-consolidated rows are skipped");

        let conn = e.store.conn.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT count(*) FROM episodic_memory", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
        let logged: i64 = conn
            .query_row(
                "SELECT count(*) FROM consolidation_log WHERE items_consolidated = 2",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(logged, 1);
    }

    #[test]
    fn episodic_recall_after_consolidation_dedups_cross_tier() {
        let e = engine();
        e.remember("the deployment uses a blue-green rollout", &RememberArgs::default())
            .unwrap();
        e.consolidate().unwrap();

        // The content now lives in BOTH tiers; recall must surface it exactly once (cross-tier dedup).
        let hits = e.recall("deployment rollout", 5).unwrap();
        let matches: Vec<_> = hits
            .iter()
            .filter(|h| h.content.contains("blue-green"))
            .collect();
        assert_eq!(matches.len(), 1, "cross-tier duplicate collapsed to one row");
    }

    #[test]
    fn episodic_vector_recall_uses_binary_and_cosine() {
        // Promote two memories with stored embeddings (consolidate also packs MIB binary vectors),
        // then recall by a query vector parallel to one of them with NO lexical overlap. Only the
        // episodic vector + binary path can surface it.
        let e = engine();
        let near = [0.96f32, 0.28, 0.0];
        let far = [0.0f32, 0.0, 1.0];
        e.remember_with_vector("alpha apple", &RememberArgs::default(), Some(&near), "mock")
            .unwrap();
        e.remember_with_vector("beta banana", &RememberArgs::default(), Some(&far), "mock")
            .unwrap();
        e.consolidate().unwrap();

        let q = [1.0f32, 0.0, 0.0];
        let hits = e.recall_with_vector("zzz", 5, Some(&q)).unwrap();
        assert!(
            hits.iter().any(|h| h.content == "alpha apple"),
            "episodic vector recall should surface the semantically-close memory"
        );
        assert!(
            hits.iter().all(|h| h.content != "beta banana"),
            "the orthogonal memory must not pass the vector gate"
        );
    }

    #[test]
    fn remember_extracts_entities_and_facts() {
        let e = engine();
        let id = e
            .remember("Maya works at Acme and uses Postgres", &RememberArgs::default())
            .unwrap();
        let c = e.store.conn.lock().unwrap();

        // Entities became `mentions` annotations.
        let mentions = annotations::query_by_memory(&c, &id, Some("mentions")).unwrap();
        assert!(
            mentions.iter().any(|m| m.value == "Maya"),
            "expected a Maya mention, got {mentions:?}"
        );

        // SPO triples landed in `facts` and were consolidated.
        let fact_rows: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM facts WHERE source_msg_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(fact_rows >= 1, "expected at least one extracted fact");
        let works_at: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM consolidated_facts \
                 WHERE subject = 'Maya' AND predicate = 'works_at' AND object = 'Acme'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(works_at, 1, "Maya works_at Acme should be consolidated");
    }

    #[test]
    fn entity_and_fact_match_reorders_recall() {
        let e = engine();
        // The entity-/fact-bearing memory (capitalized "Acme" -> entity + `works_at` fact)...
        e.remember("Maya works at Acme on infrastructure", &RememberArgs::default())
            .unwrap();
        // ...and a lexical-only distractor that mentions "acme" lowercase (no entity extracted).
        e.remember("the acme deadline is approaching", &RememberArgs::default())
            .unwrap();

        // A capitalized-entity query: both rows match lexically, but the entity/fact multipliers
        // must lift the structured memory to the top.
        let hits = e.recall("Acme", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(
            hits[0].content.contains("Maya"),
            "entity+fact match should rank first, got {:?}",
            hits.iter().map(|h| (&h.content, h.score)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cooccurrence_links_memories_sharing_an_entity() {
        let e = engine();
        let a = e
            .remember("Maya leads the Phoenix team", &RememberArgs::default())
            .unwrap();
        let b = e
            .remember("Maya approved the Phoenix budget", &RememberArgs::default())
            .unwrap();
        let c = e.store.conn.lock().unwrap();
        // The two memories share the "Maya"/"Phoenix" entities -> a `references` edge was drawn.
        assert!(episodic_graph::edge_count(&c, &a).unwrap() >= 1);
        let related = episodic_graph::find_related_memories(&c, &a, 2, "", 0.0).unwrap();
        assert!(
            related.iter().any(|r| r.memory_id == b),
            "graph should relate the two Maya/Phoenix memories"
        );
    }

    #[test]
    fn ingest_extracted_merges_llm_entities_and_triples() {
        let e = engine();
        let id = e
            .remember("a routine note", &RememberArgs::default())
            .unwrap();
        let extracted = crate::extract::Extracted {
            entities: vec!["Denis".into()],
            triples: vec![crate::extract::ExtractedTriple {
                subject: "Denis".into(),
                predicate: "manages".into(),
                object: "Atlas".into(),
                confidence: 0.9,
            }],
            facts: vec!["Denis manages the Atlas project".into()],
        };
        e.ingest_extracted(&id, &extracted).unwrap();
        let c = e.store.conn.lock().unwrap();
        let mentions: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1 AND kind = 'mentions' AND value = 'Denis'",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mentions, 1, "LLM entity should land as a mention");
        let triple: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM consolidated_facts WHERE subject='Denis' AND predicate='manages' AND object='Atlas'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(triple, 1, "LLM triple should be consolidated");
        let fact_ann: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1 AND kind = 'fact'",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(fact_ann, 1, "LLM statement should land as a fact annotation");
    }

    #[test]
    fn temporal_columns_populated_on_write() {
        let e = engine();
        let id = e
            .remember("ship the release on 2026-05-20", &RememberArgs::default())
            .unwrap();
        let c = e.store.conn.lock().unwrap();
        let (date, precision): (Option<String>, String) = c
            .query_row(
                "SELECT event_date, event_date_precision FROM working_memory WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(date.as_deref(), Some("2026-05-20"));
        assert_eq!(precision, "day");
    }

    #[test]
    fn sleep_groups_and_summarizes_with_aaak() {
        let e = engine();
        // Two rows from the same source -> one summary group.
        e.remember("User prefers dark mode", &RememberArgs::default())
            .unwrap();
        e.remember("User prefers tabs over spaces", &RememberArgs::default())
            .unwrap();
        let report = e.sleep(true).expect("forced sleep");
        assert_eq!(report.items_consolidated, 2);
        assert_eq!(report.summaries_created, 1);
        assert_eq!(report.llm_used, 0, "no LLM -> AAAK fallback");
        // A summary episodic row was written, tagged as a sleep consolidation.
        let c = e.store.conn.lock().unwrap();
        let summaries: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM episodic_memory WHERE source = 'sleep_consolidation'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(summaries, 1);
        // The originals are marked consolidated (additive: still present).
        let pending: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pending, 0, "all working rows claimed");
    }

    #[test]
    fn sleep_skips_pinned_and_respects_cutoff() {
        let e = engine();
        let id = e
            .remember("recent unpinned note", &RememberArgs::default())
            .unwrap();
        {
            let c = e.store.conn.lock().unwrap();
            c.execute("UPDATE working_memory SET pinned = 1 WHERE id = ?1", params![id])
                .unwrap();
        }
        // force=false: the row is fresh (after the cutoff) AND pinned -> nothing consolidates.
        let report = e.sleep(false).expect("sleep");
        assert_eq!(report.items_consolidated, 0);
    }

    #[test]
    fn degrade_episodic_promotes_old_tiers() {
        let e = engine();
        // Seed an episodic row backdated > TIER2_DAYS so tier1->2 fires.
        {
            let c = e.store.conn.lock().unwrap();
            c.execute(
                "INSERT INTO episodic_memory (id, content, session_id, tier, created_at) \
                 VALUES ('old1', 'User prefers Python and Rust over Go', 'default', 1, \
                         datetime('now', '-60 days'))",
                [],
            )
            .unwrap();
            // And one backdated > TIER3_DAYS at tier 2 so tier2->3 fires.
            let long = "x ".repeat(400);
            c.execute(
                "INSERT INTO episodic_memory (id, content, session_id, tier, created_at) \
                 VALUES ('old2', ?1, 'default', 2, datetime('now', '-200 days'))",
                params![long],
            )
            .unwrap();
        }
        let (t1, t2) = e.degrade_episodic().expect("degrade");
        assert_eq!(t1, 1, "tier1 row should promote to tier2");
        assert_eq!(t2, 1, "tier2 row should promote to tier3");
        let c = e.store.conn.lock().unwrap();
        let tier1: i64 = c
            .query_row("SELECT tier FROM episodic_memory WHERE id='old1'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tier1, 2);
        let (tier2, len): (i64, i64) = c
            .query_row(
                "SELECT tier, LENGTH(content) FROM episodic_memory WHERE id='old2'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(tier2, 3);
        assert!(len as usize <= TIER3_MAX_CHARS + 8, "tier3 content compressed");
    }

    #[test]
    fn tool_backing_methods_round_trip() {
        let e = engine();
        let id = e.remember("a fact to manage", &RememberArgs::default()).unwrap();
        assert!(e.get(&id).unwrap().is_some());
        assert!(e.update(&id, Some("an updated fact"), Some(0.9)).unwrap());
        assert_eq!(e.get(&id).unwrap().unwrap().content, "an updated fact");

        // Scratchpad CRUD.
        e.scratchpad_write("remember to ship").unwrap();
        assert_eq!(e.scratchpad_read().unwrap().len(), 1);
        assert_eq!(e.scratchpad_clear().unwrap(), 1);
        assert!(e.scratchpad_read().unwrap().is_empty());

        // Triples + canonical.
        e.triple_add("Ada", "uses", "Rust", None, None, "tool", 1.0, true)
            .unwrap();
        assert_eq!(e.triple_query(Some("Ada"), None, None, None).unwrap().len(), 1);
        let (_row, status) = e
            .canonical_remember("ada", "identity", "lang", "Rust", "tool", 1.0)
            .unwrap();
        assert_eq!(status, crate::knowledge::canonical::Status::Created);
        assert_eq!(e.canonical_recall("ada", None, None).unwrap().len(), 1);

        // Invalidate drops it from recall surface.
        assert!(e.invalidate(&id, None).unwrap());
        assert!(e.get(&id).unwrap().is_none());

        // Forget hard-deletes.
        let id2 = e.remember("ephemeral", &RememberArgs::default()).unwrap();
        assert!(e.forget(&id2).unwrap());
        assert!(!e.forget(&id2).unwrap(), "already gone");
    }

    #[test]
    fn export_import_round_trips_rows() {
        let e = engine();
        e.remember("portable memory one", &RememberArgs::default()).unwrap();
        e.remember("portable memory two", &RememberArgs::default()).unwrap();
        let bundle = e.export().unwrap();

        let e2 = engine();
        let n = e2.import(&bundle).unwrap();
        assert_eq!(n, 2, "both working rows imported");
        assert!(!e2.recall("portable memory", 5).unwrap().is_empty());
    }

    #[test]
    fn stats_and_diagnose_report_counts() {
        let e = engine();
        e.remember("count me", &RememberArgs::default()).unwrap();
        let stats = e.stats().unwrap();
        assert_eq!(stats.working, 1);
        let diag = e.diagnose().unwrap();
        assert_eq!(diag.pending_consolidation, 1);
    }

    #[test]
    fn enhanced_recall_uses_synonym_expansion() {
        // Enhanced recall expands "db" -> the `database` synonym group, so a query that shares no
        // surface token with the stored row still surfaces it (base recall alone would miss "db").
        let cfg = MnemosyneConfig { recall_mode: RecallMode::Enhanced, ..MnemosyneConfig::default() };
        let e = Engine::open_in_memory(cfg).unwrap();
        e.remember("the database password rotation is monthly", &RememberArgs::default())
            .unwrap();
        e.remember("lunch was margherita pizza", &RememberArgs::default()).unwrap();

        let hits = e.recall("db password", 5).unwrap();
        assert!(!hits.is_empty(), "enhanced recall should surface via synonym expansion");
        assert!(hits[0].content.contains("password"), "got: {}", hits[0].content);
        // A second identical query is served from the cache and stays consistent.
        let again = e.recall("db password", 5).unwrap();
        assert_eq!(again[0].content, hits[0].content);
    }

    #[test]
    fn base_recall_unchanged_when_flags_off() {
        // The default (Base) mode must not synonym-expand: "db" shares no token with the row, so a
        // base recall returns nothing (proving enhanced behavior is opt-in, no base regression).
        let e = engine();
        e.remember("the database password rotation is monthly", &RememberArgs::default())
            .unwrap();
        assert!(e.recall("db", 5).unwrap().is_empty(), "base recall must not expand synonyms");
    }

    #[test]
    fn polyphonic_recall_fuses_voices() {
        let cfg =
            MnemosyneConfig { recall_mode: RecallMode::Polyphonic, ..MnemosyneConfig::default() };
        let e = Engine::open_in_memory(cfg).unwrap();
        let acme_vec = [1.0f32, 0.0, 0.0];
        e.remember_with_vector(
            "Acme is a company",
            &RememberArgs::default(),
            Some(&acme_vec),
            "mock",
        )
        .unwrap();
        e.remember_with_vector(
            "unrelated note about pizza",
            &RememberArgs::default(),
            Some(&[0.0, 1.0, 0.0]),
            "mock",
        )
        .unwrap();

        // "Acme" hits the graph/fact voices (fact subject "Acme") and the vector voice (parallel
        // query vector); RRF fusion should surface the Acme row.
        let hits = e.recall_with_vector("Acme", 5, Some(&acme_vec)).unwrap();
        assert!(hits.iter().any(|h| h.content == "Acme is a company"), "polyphonic fused result");
    }
}
