// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Engine configuration (data root, bank, session, decay/TTL knobs).
//!
//! Pure data — no environment reads. The node binary's `NodeConfig` (figment) owns all layering and
//! injects the resolved values (`data_dir`, `[mnemosyne]` recall/identity knobs) at construction.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Which recall pipeline `Engine::recall` dispatches to (`beam.py` `recall` L5098 polyphonic reroute
/// / `recall_enhanced` L6202 gate). Defaults to `Base`; the host selects it via `[mnemosyne].recall_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallMode {
    /// The base hybrid cross-tier recall (`recall`), default.
    #[default]
    Base,
    /// Intent-weighted + synonym-expanded + Weibull-rescored + MMR pipeline (`recall_enhanced`,
    /// `MNEMOSYNE_ENHANCED_RECALL=1`).
    Enhanced,
    /// Four-voice RRF recall (`_recall_polyphonic`, `MNEMOSYNE_POLYPHONIC_RECALL=1`).
    Polyphonic,
}

/// Per-label veracity score multipliers (`veracity_consolidation.py` `VERACITY_WEIGHTS` L122-L128,
/// env-overridable per label in `beam.py` L340-L345).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VeracityWeights {
    /// `stated` (default 1.0, `MNEMOSYNE_STATED_WEIGHT`).
    pub stated: f64,
    /// `inferred` (default 0.7, `MNEMOSYNE_INFERRED_WEIGHT`).
    pub inferred: f64,
    /// `tool` (default 0.5, `MNEMOSYNE_TOOL_WEIGHT`).
    pub tool: f64,
    /// `imported` (default 0.6, `MNEMOSYNE_IMPORTED_WEIGHT`).
    pub imported: f64,
    /// `unknown` (default 0.8, `MNEMOSYNE_UNKNOWN_WEIGHT`).
    pub unknown: f64,
}

impl Default for VeracityWeights {
    fn default() -> Self {
        Self {
            stated: 1.0,
            inferred: 0.7,
            tool: 0.5,
            imported: 0.6,
            unknown: 0.8,
        }
    }
}

impl VeracityWeights {
    /// The multiplier for a (clamped) veracity label; unrecognized labels weigh as `unknown`.
    pub fn weight(&self, veracity: &str) -> f64 {
        match veracity {
            "stated" => self.stated,
            "inferred" => self.inferred,
            "tool" => self.tool,
            "imported" => self.imported,
            _ => self.unknown,
        }
    }
}

/// Engine configuration.
#[derive(Clone, Debug)]
pub struct MnemosyneConfig {
    /// The data root (default `$HERMES_HOME/mnemosyne/data` or `MNEMOSYNE_DATA_DIR`).
    pub data_dir: PathBuf,
    /// The memory bank name (`default` -> `{data_dir}/mnemosyne.db`).
    pub bank: String,
    /// The active session id.
    pub session_id: String,
    /// Recency decay half-life in hours (`RECENCY_HALFLIFE_HOURS`, default 168).
    pub recency_halflife_hours: f64,
    /// Working-memory TTL in hours (`WORKING_MEMORY_TTL_HOURS`, default 168). Bounds the
    /// not-yet-consolidated trim window; sleep's consolidation age cutoff is half this.
    pub working_memory_ttl_hours: f64,
    /// Working-memory row cap per session before trim (`WORKING_MEMORY_MAX_ITEMS`,
    /// `MNEMOSYNE_WM_MAX_ITEMS`, default 10000).
    pub working_memory_max_items: usize,
    /// Cap on how often recall bumps `last_recalled` (`WM_BUMP_CAP_HOURS`, default 24).
    pub wm_bump_cap_hours: f64,
    /// Episodic fallback-scan limit (`EPISODIC_RECALL_LIMIT`, `MNEMOSYNE_EP_LIMIT`, default 50000).
    pub episodic_recall_limit: usize,
    /// Scratchpad row cap per session (`SCRATCHPAD_MAX_ITEMS`, `MNEMOSYNE_SP_MAX`, default 1000).
    pub scratchpad_max_items: usize,
    /// Max working rows claimed per sleep pass (`SLEEP_BATCH_SIZE`, default 5000).
    pub sleep_batch_size: usize,
    /// Tier 1->2 degradation age in days (`TIER2_DAYS`, default 30).
    pub tier2_days: i64,
    /// Tier 2->3 degradation age in days (`TIER3_DAYS`, default 180).
    pub tier3_days: i64,
    /// Episodic tier score weights `[T1, T2, T3]` (`MNEMOSYNE_TIER{1,2,3}_WEIGHT`,
    /// default `[1.0, 0.5, 0.25]`).
    pub tier_weights: [f64; 3],
    /// Hybrid recall weights `(vec, fts, importance)` before normalization
    /// (`MNEMOSYNE_{VEC,FTS,IMPORTANCE}_WEIGHT`, defaults 0.5/0.3/0.2, `beam.py` L1157-L1183).
    pub recall_weights: (f64, f64, f64),
    /// Apply the veracity score multiplier (A/B toggle `MNEMOSYNE_VERACITY_MULTIPLIER=0` disables;
    /// default on, `beam.py` L5950-L5972).
    pub veracity_multiplier: bool,
    /// Per-label veracity multipliers (env-overridable in Python; injected here).
    pub veracity_weights: VeracityWeights,
    /// Apply the episodic graph-edge bonus (`MNEMOSYNE_GRAPH_BONUS=0` disables).
    pub graph_bonus: bool,
    /// Apply the episodic fact-match bonus (`MNEMOSYNE_FACT_BONUS=0` disables).
    pub fact_bonus: bool,
    /// Apply the MIB binary-vector bonus (`MNEMOSYNE_BINARY_BONUS=0` disables).
    pub binary_bonus: bool,
    /// Cross-tier summary dedup in recall finalize (`MNEMOSYNE_CROSS_TIER_DEDUP=0` disables).
    pub cross_tier_dedup: bool,
    /// Merge structured `fact_recall` hits into recall output (`MNEMOSYNE_FACT_RECALL_ENABLED=1`
    /// enables; default off, `beam.py` L6152).
    pub fact_recall_enabled: bool,
    /// Lenient fact-aware recall matching (`MNEMOSYNE_LENIENT_FACT_MATCH=1`; default strict,
    /// `beam.py` L1703).
    pub lenient_fact_match: bool,
    /// Default temporal-boost half-life in hours (`MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS`, default 24).
    pub temporal_halflife_hours: f64,
    /// Proactive graph linking at ingest (`MNEMOSYNE_PROACTIVE_LINKING=1` enables; default off,
    /// `beam.py` `_proactively_link` L3358).
    pub proactive_linking: bool,
    /// Which recall pipeline to use (default [`RecallMode::Base`]).
    pub recall_mode: RecallMode,
    /// Enable the opt-in tier-2 LLM conflict detector in sleep (`MNEMOSYNE_LLM_CONFLICT_DETECTION`).
    pub llm_conflict_detection: bool,
    /// Multi-agent identity: the original writer (`MNEMOSYNE_AUTHOR_ID`, `beam.py` ctor L2616). When
    /// set it both stamps new rows and widens recall scope to all sessions for that author.
    pub author_id: Option<String>,
    /// Multi-agent identity: author type — `human`/`agent`/`system` (`MNEMOSYNE_AUTHOR_TYPE`).
    pub author_type: Option<String>,
    /// Multi-agent identity: channel/group id (`MNEMOSYNE_CHANNEL_ID`, `beam.py` ctor L2618). Writes
    /// default it to `session_id`; recall only filters on it when explicitly set.
    pub channel_id: Option<String>,
    /// Named prefetch profile for the provider's per-turn recall injection
    /// (`MNEMOSYNE_PREFETCH_PROFILE`; `general` or `social-chat`, unknown -> `general`).
    pub prefetch_profile: String,
    /// Per-memory prefetch content char cap; `0` = untruncated
    /// (`MNEMOSYNE_PREFETCH_CONTENT_CHARS`, default 0).
    pub prefetch_content_chars: usize,
    /// Auto-run a sleep pass from `after_turn` every 10 persisted turns when working memory
    /// exceeds [`Self::auto_sleep_threshold`] (`MNEMOSYNE_AUTO_SLEEP_ENABLED`, default off).
    pub auto_sleep_enabled: bool,
    /// Working-memory row count that arms auto-sleep (`sleep_threshold`, default 50).
    pub auto_sleep_threshold: usize,
    /// Conversation roles persisted by `after_turn` (`MNEMOSYNE_SYNC_ROLES` / `sync_roles`,
    /// default `["user", "assistant"]`; empty disables autosave; identity capture is gated by
    /// `user`).
    pub sync_roles: Vec<String>,
    /// Regex patterns filtered from turn autosave (`ignore_patterns`, case-insensitive search;
    /// invalid patterns are skipped).
    pub ignore_patterns: Vec<String>,
    /// Directory holding the shared-surface bank (`shared_surface_path`; the DB file is
    /// `<dir>/mnemosyne.db`). `None` -> `<data_dir>/shared`.
    pub shared_surface_dir: Option<PathBuf>,
    /// Merge shared-surface hits into `mnemosyne_recall`, tagging each result's `bank`
    /// (`shared_surface_read`, default off).
    pub shared_surface_read: bool,
    /// Remote sync server URL for the `mnemosyne_sync_*` tools (`MNEMOSYNE_SYNC_REMOTE` /
    /// `MNEMOSYNE_SYNC_HOST`+`PORT`; `sync` feature). `None` = replication unconfigured.
    pub sync_remote: Option<String>,
    /// Bearer token sent to the remote sync server (`MNEMOSYNE_SYNC_TOKEN`).
    pub sync_token: Option<String>,
    /// Sync payload encryption key source: a raw urlsafe-base64 32-byte key or a key-file path
    /// (`MNEMOSYNE_SYNC_KEY` / `MNEMOSYNE_SYNC_KEY_SOURCE=file:<path>`). `None` = plaintext.
    pub sync_key: Option<String>,
    /// Sync direction for the replication cycle (`MNEMOSYNE_SYNC_MODE`): `bidirectional`
    /// (default), `push`, or `pull`.
    pub sync_mode: String,
}

/// The multi-agent identity scope applied to a recall (`beam.py` `recall` author/channel params
/// L5030-L5032, scope clause L5182-L5220). All-`None` reproduces the default
/// `session_id OR scope='global'` behavior.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecallScope {
    /// Filter to a specific author (and, with no channel, widen to all sessions for that author).
    pub author_id: Option<String>,
    /// Filter to a specific author type.
    pub author_type: Option<String>,
    /// Filter to a specific channel/group (widens recall to `session OR global OR channel`).
    pub channel_id: Option<String>,
}

impl RecallScope {
    /// Whether any identity filter is set (an empty scope keeps the default session behavior).
    pub fn is_empty(&self) -> bool {
        self.author_id.is_none() && self.author_type.is_none() && self.channel_id.is_none()
    }
}

/// Per-call recall filters and temporal scoring knobs (`beam.py` `recall` kwargs L5027-L5040):
/// date-range / source / topic / veracity / memory-type row filters plus the Phase-3 soft
/// temporal boost. [`Default`] disables everything (the plain recall behavior).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecallFilters {
    /// Lower timestamp bound, `YYYY-MM-DD` (expanded to `T00:00:00`).
    pub from_date: Option<String>,
    /// Upper timestamp bound, `YYYY-MM-DD` (expanded to `T23:59:59`).
    pub to_date: Option<String>,
    /// Exact `source` column match.
    pub source: Option<String>,
    /// Topic tag; stored in `source` pending a dedicated column (`beam.py` L5202).
    pub topic: Option<String>,
    /// Exact `veracity` label match.
    pub veracity: Option<String>,
    /// Exact `memory_type` match.
    pub memory_type: Option<String>,
    /// Soft boost weight `[0, 1]` for memories near `query_time`; `0.0` disables (default).
    pub temporal_weight: f64,
    /// Temporal-boost target instant (ISO-8601); `None` = now.
    pub query_time: Option<String>,
    /// Temporal decay half-life in hours; `None` = the configured
    /// [`MnemosyneConfig::temporal_halflife_hours`].
    pub temporal_halflife: Option<f64>,
    /// Per-call vec-weight override before normalization (`beam.py` `recall(vec_weight=...)`);
    /// `None` = the configured [`MnemosyneConfig::recall_weights`]. Base pipeline only.
    pub vec_weight: Option<f64>,
    /// Per-call FTS-weight override (`fts_weight=...`).
    pub fts_weight: Option<f64>,
    /// Per-call importance-weight override (`importance_weight=...`).
    pub importance_weight: Option<f64>,
}

impl Default for MnemosyneConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("mnemosyne-data"),
            bank: "default".to_string(),
            session_id: "default".to_string(),
            recency_halflife_hours: 168.0,
            working_memory_ttl_hours: 168.0,
            working_memory_max_items: 10_000,
            wm_bump_cap_hours: 24.0,
            episodic_recall_limit: 50_000,
            scratchpad_max_items: 1000,
            sleep_batch_size: 5000,
            tier2_days: 30,
            tier3_days: 180,
            tier_weights: [1.0, 0.5, 0.25],
            recall_weights: (0.5, 0.3, 0.2),
            veracity_multiplier: true,
            veracity_weights: VeracityWeights::default(),
            graph_bonus: true,
            fact_bonus: true,
            binary_bonus: true,
            cross_tier_dedup: true,
            fact_recall_enabled: false,
            lenient_fact_match: false,
            temporal_halflife_hours: 24.0,
            proactive_linking: false,
            recall_mode: RecallMode::Base,
            llm_conflict_detection: false,
            author_id: None,
            author_type: None,
            channel_id: None,
            prefetch_profile: "general".to_string(),
            prefetch_content_chars: 0,
            auto_sleep_enabled: false,
            auto_sleep_threshold: 50,
            sync_roles: vec!["user".to_string(), "assistant".to_string()],
            ignore_patterns: Vec::new(),
            shared_surface_dir: None,
            shared_surface_read: false,
            sync_remote: None,
            sync_token: None,
            sync_key: None,
            sync_mode: "bidirectional".to_string(),
        }
    }
}

impl MnemosyneConfig {
    /// The content-addressed blob root for the sanitizer's externalized payloads
    /// (`<data_dir>/blobs`). Threaded into [`crate::sanitize::sanitize_content`] so blob storage
    /// follows the injected data dir rather than a process-global env var.
    pub fn blob_dir(&self) -> PathBuf {
        self.data_dir.join("blobs")
    }

    /// The resolved SQLite path for the configured bank (`banks.py` `get_bank_db_path`).
    pub fn bank_db_path(&self) -> PathBuf {
        if self.bank == "default" {
            self.data_dir.join("mnemosyne.db")
        } else {
            self.data_dir
                .join("banks")
                .join(&self.bank)
                .join("mnemosyne.db")
        }
    }

    /// The shared-surface bank directory (`shared_surface_path` config key; Python default
    /// `<mnemosyne>/data/shared/mnemosyne.db`). Divergence: Rust configures the *directory*, the
    /// DB file inside is always `mnemosyne.db`.
    pub fn shared_surface_dir(&self) -> PathBuf {
        self.shared_surface_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("shared"))
    }
}
