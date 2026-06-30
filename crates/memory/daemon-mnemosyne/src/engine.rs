// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The BEAM engine facade — port of `beam.py` `BeamMemory` (the `remember`/`recall`/`get_context`/
//! `sleep` surface, L2836 / L5027 / L3526 / L7576) plus the `memory.py` facade.
//!
//! As-built: `remember`/`get_context` plus a hybrid `recall` that gathers candidates across the
//! **working and episodic** tiers from FTS5 (`fts_working`/`fts_episodes`, BM25), the stored
//! embeddings (cosine), and a recency/importance fallback scan, then scores them
//! ([`crate::recall::scoring`]) with the FTS-blended lexical relevance, vector similarity, the MIB
//! `binary_bonus`, and the tier/veracity multipliers — merged, content-deduped, and MMR-diversified.
//! [`Engine::consolidate`] is a minimal WM->episodic promotion (no LLM summarization/degradation).
//! Knowledge ingestion (graph/fact bonuses) and full `sleep` remain port-spec P1 work.

use crate::config::{MnemosyneConfig, RecallScope};
use crate::error::Result;
use crate::recall::query_cache::QueryCache;
use crate::store::Store;
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

/// A heuristic sleep-time conflict: an older memory that a newer, near-identical one supersedes
/// (`beam.py` `_detect_conflicts` L3634). The older id is invalidated with the newer as replacement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepConflict {
    /// The older memory id (flagged superseded).
    pub older_id: String,
    /// The newer memory id (the replacement).
    pub newer_id: String,
    /// The older memory content (for optional LLM validation).
    pub older_content: String,
    /// The newer memory content (for optional LLM validation).
    pub newer_content: String,
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

// ── Parameter bundles (W-MNEMO: clear the pervasive Excess-Number-of-Function-Arguments smell) ──

/// Shared per-recall request parameters threaded through the recall pipeline.
pub struct RecallReq<'a> {
    /// The query text.
    pub query: &'a str,
    /// The maximum number of rows to return.
    pub top_k: usize,
    /// An optional precomputed query embedding (enables the vector source).
    pub query_vector: Option<&'a [f32]>,
    /// The multi-agent recall scope.
    pub scope: &'a RecallScope,
}

/// Per-tier candidate-gathering context for [`Engine::gather_working`] / [`Engine::gather_episodic`]:
/// the tokenized/entity-extracted query, the candidate caps, the lexical floor, the query vector,
/// the `(vec, fts, importance)` blend weights, and the recall scope.
pub(crate) struct GatherCtx<'a> {
    pub q_tokens: &'a [String],
    pub q_entities: &'a [String],
    pub top_k: usize,
    pub floor: f64,
    pub query_vector: Option<&'a [f32]>,
    pub weights: (f64, f64, f64),
    pub scope: &'a RecallScope,
}

/// A freshly-stored memory to run deterministic knowledge ingestion over ([`Engine::ingest_knowledge`]).
pub(crate) struct IngestItem<'a> {
    pub memory_id: &'a str,
    pub content: &'a str,
    pub veracity: &'a str,
}

/// Entity-seeded candidate-injection context for [`Engine::inject_entity_candidates`].
pub(crate) struct EntityInjectCtx<'a> {
    pub q_entities: &'a [String],
    pub present: &'a std::collections::HashSet<String>,
    pub scope: &'a RecallScope,
}

/// Arguments for [`Engine::triple_add`] — a temporal-triple upsert.
pub struct TripleAdd<'a> {
    /// The triple subject.
    pub subject: &'a str,
    /// The triple predicate.
    pub predicate: &'a str,
    /// The triple object.
    pub object: &'a str,
    /// Optional validity start (ISO-8601).
    pub valid_from: Option<&'a str>,
    /// Optional validity end (ISO-8601).
    pub valid_until: Option<&'a str>,
    /// Provenance source label.
    pub source: &'a str,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
    /// Whether to supersede a prior open `(subject, predicate)` triple.
    pub supersede: bool,
}

/// Arguments for [`Engine::triple_end`] — expire open triples.
pub struct TripleEnd<'a> {
    /// The triple subject.
    pub subject: &'a str,
    /// The triple predicate.
    pub predicate: &'a str,
    /// Optional object filter.
    pub object: Option<&'a str>,
    /// Optional explicit expiry (ISO-8601); defaults to now.
    pub valid_until: Option<&'a str>,
}

/// Arguments for [`Engine::triple_query`] — query triples valid at `as_of`.
pub struct TripleQuery<'a> {
    /// Optional subject filter.
    pub subject: Option<&'a str>,
    /// Optional predicate filter.
    pub predicate: Option<&'a str>,
    /// Optional object filter.
    pub object: Option<&'a str>,
    /// Optional as-of instant (ISO-8601); defaults to now.
    pub as_of: Option<&'a str>,
}

/// Arguments for [`Engine::canonical_remember`] — upsert a canonical identity fact.
pub struct CanonicalRemember<'a> {
    /// The fact owner.
    pub owner_id: &'a str,
    /// The fact category.
    pub category: &'a str,
    /// The slot name.
    pub name: &'a str,
    /// The fact body.
    pub body: &'a str,
    /// Provenance source label.
    pub source: &'a str,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
}

/// Arguments for [`Engine::validate`] — record a human/agent validation action on a memory.
pub struct ValidateArgs<'a> {
    /// The target memory id.
    pub id: &'a str,
    /// The action (`confirm` / `correct` / `reject`).
    pub action: &'a str,
    /// Optional validator identity.
    pub validator: Option<&'a str>,
    /// Replacement content (for `correct`).
    pub new_content: Option<&'a str>,
    /// Optional free-text note.
    pub note: Option<&'a str>,
}

/// Arguments for [`Engine::graph_link`] — add a manual graph edge.
pub struct GraphLink<'a> {
    /// The edge source memory id.
    pub source: &'a str,
    /// The edge target memory id.
    pub target: &'a str,
    /// The edge type (defaults to `related_to` when empty).
    pub edge_type: &'a str,
    /// The edge weight.
    pub weight: f64,
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
}

mod consolidation;
mod ingest;
mod knowledge;
mod maintenance;
mod query;
mod recall;
#[cfg(test)]
mod tests;

#[cfg(all(feature = "vec-ext", test))]
pub(crate) use query::native_cosine_sim_map;
pub(crate) use query::{cosine_sim_map, load_embeddings};
pub(crate) use recall::age_hours;
#[cfg(test)]
pub(crate) use recall::lexical_relevance;
