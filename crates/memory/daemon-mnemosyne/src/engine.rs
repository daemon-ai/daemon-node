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
//! Recall dispatches on [`crate::config::RecallMode`] (base / enhanced / polyphonic), the knowledge
//! layer (graph/fact bonuses) is wired into ingest, and `sleep` is the full plan/claim/summarize/
//! degrade pipeline ([`Engine::sleep_plan`] / [`Engine::finish_sleep`]).

use crate::config::{MnemosyneConfig, RecallFilters, RecallScope};
use crate::error::Result;
use crate::recall::diagnostics::RecallDiagnostics;
use crate::recall::query_cache::QueryCache;
use crate::store::Store;
use std::sync::OnceLock;

/// The vector-similarity floor that lets a vector-only hit survive the lexical gate (mirrors the
/// episodic candidate-drop rule `lexical < floor && sim < 0.65 -> drop`, `beam.py` L5720+).
const VEC_SIM_FLOOR: f64 = 0.65;

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

impl SleepGroup {
    /// The deterministic no-LLM summary for this group: `[source] <aaak_encode>` (`beam.py` sleep
    /// AAAK fallback L7784). Shared by [`Engine::finish_sleep`]'s fallback and the async seam (which
    /// must know the final text up front to embed it).
    pub fn aaak_summary(&self) -> String {
        format!(
            "[{}] {}",
            self.source,
            crate::aaak::summarize_group(&self.contents)
        )
    }
}

/// One sleep group's summarization outcome, produced at the async seam (`tools::run_sleep`) and
/// keyed by group `source` in [`Engine::finish_sleep`]. Python's `consolidate_to_episodic`
/// (`beam.py` L3956-L4032) embeds the summary inline; the synchronous Rust engine instead receives
/// the precomputed embedding of the **final** summary text alongside it.
#[derive(Clone, Debug, Default)]
pub struct GroupSummary {
    /// The final summary text (LLM output or the AAAK fallback), `<think>`-stripped.
    pub text: String,
    /// Whether an LLM produced `text` (drives [`SleepReport::llm_used`]).
    pub llm: bool,
    /// A dense embedding of `text`, persisted to `memory_embeddings` and binarized into
    /// `episodic_memory.binary_vector` (`beam.py` L4005-L4032). `None` in keyword-only mode.
    pub embedding: Option<Vec<f32>>,
    /// The embedding model tag for `memory_embeddings.model`.
    pub model: String,
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
    /// Well-attested `(subject, predicate)` fact conflicts auto-resolved by
    /// [`crate::knowledge::veracity::run_consolidation_pass`] at the end of the pass. (Python
    /// defines the pass but never calls it — `veracity_consolidation.py` L777; the port wires it
    /// into sleep so consolidation actually converges contested facts.)
    pub facts_auto_resolved: usize,
}

/// Which BEAM tier a recall row came from (`beam.py` result `tier` labels).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Hot, recent, auto-injected context.
    #[default]
    Working,
    /// Long-term consolidated memory.
    Episodic,
    /// The MEMORIA structured-fact supplement row (`beam.py` L6023).
    Memoria,
    /// A working row surfaced as a MEMORIA fact source (`beam.py` L6049).
    MemoriaSource,
    /// A structured `fact_recall` row merged into recall output (`beam.py` L6167).
    Fact,
}

/// A recalled / stored memory row (the `recall` result dict shape, `beam.py` L5344-L5366).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MemoryRow {
    /// Memory id.
    pub id: String,
    /// Content text (recall truncates to 500 chars, `beam.py` L5346).
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
    /// The episodic degradation tier (`1`/`2`/`3`); working rows are always `1` (`beam.py` L5960).
    pub tier_level: i64,
    /// The recall score (0 for direct fetches), rounded to 4 places.
    pub score: f64,
    /// The lexical relevance signal that fed the score (`keyword_score`).
    #[serde(default)]
    pub keyword_score: f64,
    /// The dense/vector similarity signal (`dense_score`).
    #[serde(default)]
    pub dense_score: f64,
    /// The FTS5 signal (`fts_score`).
    #[serde(default)]
    pub fts_score: f64,
    /// The recency decay factor applied (`recency_decay`).
    #[serde(default)]
    pub recency_decay: f64,
    /// Times this row has been recalled (`recall_count`).
    #[serde(default)]
    pub recall_count: i64,
    /// When this row was last recalled (`last_recalled`).
    #[serde(default)]
    pub last_recalled: Option<String>,
    /// Row scope (`session`/`global`).
    #[serde(default)]
    pub scope: String,
    /// Multi-agent author id, when stamped.
    #[serde(default)]
    pub author_id: Option<String>,
    /// Multi-agent author type, when stamped.
    #[serde(default)]
    pub author_type: Option<String>,
    /// Multi-agent channel id, when stamped.
    #[serde(default)]
    pub channel_id: Option<String>,
    /// Row expiry; recall admits rows with `valid_until > now` (`beam.py` L5177).
    #[serde(default)]
    pub valid_until: Option<String>,
    /// Whether the entity-aware pass matched/boosted this row (`entity_match`).
    #[serde(default)]
    pub entity_match: bool,
    /// Whether the fact-aware pass matched/boosted this row (`fact_match`).
    #[serde(default)]
    pub fact_match: bool,
    /// Per-signal provenance (`voice_scores`): the linear path collapses its scoring signals into
    /// `vec/fts/keyword/importance/recency_decay` (`beam.py` L5987-L5994, Gap G); the polyphonic
    /// path carries per-voice RRF contributions (`beam.py` L6649).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_scores: Option<std::collections::HashMap<String, f64>>,
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
    /// Trust label (default `unknown`; clamped to the canonical set on write).
    pub veracity: String,
    /// Caller-supplied metadata object persisted to `metadata_json` (`beam.py` `metadata`).
    pub metadata: Option<serde_json::Value>,
    /// Expiry timestamp; the row drops out of recall past it (`beam.py` `valid_until`).
    pub valid_until: Option<String>,
    /// Pre-generated memory id; a fresh time-salted id is derived when absent
    /// (`beam.py` `memory_id` passthrough, L2843).
    pub memory_id: Option<String>,
    /// Extract regex entity `mentions` annotations (`beam.py` `extract_entities`, default off).
    pub extract_entities: bool,
    /// Request LLM fact extraction (`beam.py` `extract`, default off). The synchronous engine
    /// records the request; the async provider/tool layer honors it via
    /// [`Engine::ingest_extracted`] after the LLM round-trip.
    pub extract: bool,
    /// Explicit trust tier; derived from `source` when absent (`beam.py` `trust_tier` L152-L188).
    pub trust_tier: Option<String>,
}

impl Default for RememberArgs {
    fn default() -> Self {
        Self {
            source: "conversation".to_string(),
            importance: 0.5,
            scope: "session".to_string(),
            veracity: "unknown".to_string(),
            metadata: None,
            valid_until: None,
            memory_id: None,
            extract_entities: false,
            extract: false,
            trust_tier: None,
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
    /// Row filters + temporal scoring knobs (`beam.py` `recall` kwargs).
    pub filters: RecallFilters,
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

/// What [`Engine::validate_action`] observed before applying a collaborative-attestation action
/// (`_handle_validate`'s response fields, `hermes_memory_provider/__init__.py` L2201-L2207).
#[derive(Clone, Debug)]
pub struct ValidationOutcome {
    /// The row's original author — preserved across validations.
    pub author_id: Option<String>,
    /// The row content before the action was applied.
    pub previous_content: String,
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
    /// Stable per-bank device identity for the event log (`sync.py` L628-L640), lazily read from /
    /// persisted to `sync_meta`.
    device_id: OnceLock<String>,
    /// Monotonic per-process disambiguator for event ids minted within the same instant.
    event_seq: std::sync::atomic::AtomicU64,
    /// Recall path provenance counters (`recall_diagnostics.py`; Python's process-global
    /// singleton, owned per engine here).
    recall_diag: RecallDiagnostics,
    /// Lazily-materialized plugin manager (`memory.py` "Phase 8: Plugins" lazy property
    /// L286-L292). Never touched by a host = never built = zero overhead per event.
    plugins: OnceLock<crate::plugins::PluginManager>,
    /// Lazily-enabled in-process event stream (`memory.py` "Phase 8: Streaming" lazy `stream`
    /// property + `enable_streaming` L166-L189). Never enabled = one atomic load per write.
    stream: OnceLock<std::sync::Arc<crate::streaming::MemoryStream>>,
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
            device_id: OnceLock::new(),
            event_seq: std::sync::atomic::AtomicU64::new(0),
            recall_diag: RecallDiagnostics::default(),
            plugins: OnceLock::new(),
            stream: OnceLock::new(),
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
            device_id: OnceLock::new(),
            event_seq: std::sync::atomic::AtomicU64::new(0),
            recall_diag: RecallDiagnostics::default(),
            plugins: OnceLock::new(),
            stream: OnceLock::new(),
        })
    }

    /// The recall path provenance counters (`recall_diagnostics.py` `get_diagnostics`).
    pub fn recall_diagnostics(&self) -> &RecallDiagnostics {
        &self.recall_diag
    }

    /// The plugin manager, materialized on first access (`memory.py` lazy `plugins` property
    /// L286-L292). Built-ins are registered but unloaded; call
    /// [`crate::plugins::PluginManager::load_all`] or load individually to activate them.
    pub fn plugins(&self) -> &crate::plugins::PluginManager {
        self.plugins.get_or_init(crate::plugins::PluginManager::new)
    }

    /// The plugin manager only if a host has materialized it — the lifecycle notification
    /// call sites use this so an untouched manager costs one atomic load per event.
    pub(crate) fn plugins_if_active(&self) -> Option<&crate::plugins::PluginManager> {
        self.plugins.get()
    }

    /// Enable event streaming for this engine (`memory.py` `enable_streaming` L174-L184): wires
    /// a [`crate::streaming::MemoryStream`] into the write path so mutations emit
    /// `MEMORY_ADDED`/`MEMORY_UPDATED`/`MEMORY_CONSOLIDATED` events. Idempotent — repeat calls
    /// return the same stream. Streaming failures never block memory operations.
    pub fn enable_streaming(&self) -> std::sync::Arc<crate::streaming::MemoryStream> {
        self.stream
            .get_or_init(|| std::sync::Arc::new(crate::streaming::MemoryStream::default()))
            .clone()
    }

    /// The stream only if a host has enabled it (`beam.py` `_event_emitter is None` gate L2817).
    pub(crate) fn stream_if_active(&self) -> Option<&crate::streaming::MemoryStream> {
        self.stream.get().map(std::sync::Arc::as_ref)
    }

    /// Run `f` against the bank connection. The seam for the sibling sync/streaming modules
    /// ([`crate::streaming::DeltaSync`], `sync::SyncEngine`) which own their SQL but not the
    /// connection (Python hands them `self.conn` directly).
    pub(crate) fn with_conn<T>(
        &self,
        f: impl FnOnce(&rusqlite::Connection) -> Result<T>,
    ) -> Result<T> {
        let conn = self.store.conn.lock().unwrap();
        f(&conn)
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

    /// The engine configuration (read-only; the provider/tool layer reads its knobs).
    pub fn config(&self) -> &MnemosyneConfig {
        &self.config
    }

    /// Whether this engine is backed by a disk file (false for the ephemeral in-memory banks).
    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    /// Whether the opt-in tier-2 LLM conflict detector is enabled (`MNEMOSYNE_LLM_CONFLICT_DETECTION`).
    pub fn llm_conflict_detection(&self) -> bool {
        self.config.llm_conflict_detection
    }

    // ── SHMR (opt-in background pass; `shmr.py`) ────────────────────────────────────────────────

    /// Run one SHMR harmonic cycle over recent memories (`shmr.py` `harmonize`). Off the hot path
    /// — Python never wires it into `sleep()` — and the engine owns no LLM/embedding runtime, so
    /// both are injected ([`crate::recall::shmr`]).
    pub fn harmonize(
        &self,
        opts: &crate::recall::shmr::ShmrOptions,
        embed: crate::recall::shmr::EmbedFn,
        llm: crate::recall::shmr::LlmFn,
    ) -> Result<crate::recall::shmr::HarmonizeStats> {
        let conn = self.store.conn.lock().unwrap();
        crate::recall::shmr::harmonize(&conn, &self.config.session_id, opts, embed, llm)
    }

    /// Search `harmonic_beliefs` for a query (`shmr.py` `recall_beliefs`).
    pub fn recall_beliefs(
        &self,
        query: &str,
        top_k: usize,
        embed: crate::recall::shmr::EmbedFn,
    ) -> Result<Vec<crate::recall::shmr::BeliefHit>> {
        let conn = self.store.conn.lock().unwrap();
        crate::recall::shmr::recall_beliefs(&conn, query, top_k, embed)
    }

    /// Phase-3A reflective synthesis over [`Engine::fact_recall`] hits (`shmr.py` `reflect`).
    pub fn reflect(
        &self,
        question: &str,
        top_k: usize,
        llm: crate::recall::shmr::LlmFn,
    ) -> Result<Option<String>> {
        let facts = self.fact_recall(question, top_k)?;
        Ok(crate::recall::shmr::reflect(question, &facts, top_k, llm))
    }

    /// Recent harmonization run logs (`shmr.py` `get_resonance_log`).
    pub fn resonance_log(&self, limit: usize) -> Result<Vec<crate::recall::shmr::ResonanceEntry>> {
        let conn = self.store.conn.lock().unwrap();
        crate::recall::shmr::get_resonance_log(&conn, limit)
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

#[cfg(feature = "sync")]
pub(crate) use ingest::suppress_event_log;
pub(crate) use query::load_embeddings;
#[cfg(all(feature = "vec-ext", test))]
pub(crate) use query::native_cosine_sim_map;
pub use recall::FactHit;
