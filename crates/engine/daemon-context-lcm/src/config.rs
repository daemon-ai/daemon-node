// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Configuration for the LCM context engine (`daemon-context-lcm-port-spec.md` §12.1).
//!
//! The full `LCM:config.py` field table: the M1-M4 compaction fields, the M5 (protection §8/§9)
//! and M7 (filters/routing/presets §12.3-12.6) knobs, and the opt-in escape hatches (dynamic leaf
//! chunking, cache-friendly condensation, deferred-maintenance debt, critical-pressure bypass,
//! assembly caps). Constants carry the Python defaults verbatim (Appendix A). Every opt-in
//! defaults *off* (empty pattern lists / disabled flags / `0` caps), so the zero-config default
//! path is the always-on compaction path only. The `*_source` fields are pure provenance
//! diagnostics for `lcm_status` (`default` until the host injects an override — the daemon
//! equivalent of Python's `env`).

use std::path::{Path, PathBuf};

/// The fraction of the model context window at which compaction triggers (Appendix A).
const DEFAULT_CONTEXT_THRESHOLD: f64 = 0.35;
/// The number of most-recent turns always kept verbatim (Appendix A).
const DEFAULT_FRESH_TAIL_COUNT: usize = 32;
/// The base leaf-chunk size in tokens (Appendix A).
const DEFAULT_LEAF_CHUNK_TOKENS: usize = 20_000;
/// The number of sibling nodes that triggers a condensation to the next depth (Appendix A).
const DEFAULT_CONDENSATION_FANIN: usize = 4;
/// The maximum condensation depth (`0` disables, `-1` unlimited) (Appendix A).
const DEFAULT_INCREMENTAL_MAX_DEPTH: i64 = 3;
/// The dynamic leaf-chunk working-threshold ceiling in tokens (`LCM:config.py:220`).
const DEFAULT_DYNAMIC_LEAF_CHUNK_MAX: usize = 40_000;
/// The same-depth fanin-group floor for one cache-friendly follow-on condensation
/// (`LCM:config.py:226`).
const DEFAULT_CACHE_FRIENDLY_MIN_DEBT_GROUPS: usize = 2;
/// The extra leaf passes a debt-triggered maintenance turn may spend (`LCM:config.py:232`).
const DEFAULT_DEFERRED_MAINTENANCE_MAX_PASSES: usize = 4;
/// The L2 budget as a fraction of the L1 budget (Appendix A).
const DEFAULT_L2_BUDGET_RATIO: f64 = 0.50;
/// The deterministic L3 truncation budget in tokens (Appendix A).
const DEFAULT_L3_TRUNCATE_TOKENS: usize = 512;
/// The per-summary auxiliary-provider timeout in milliseconds (Appendix A).
const DEFAULT_SUMMARY_TIMEOUT_MS: u64 = 60_000;
/// The large-payload externalization threshold in characters (Appendix A).
const DEFAULT_EXTERNALIZATION_THRESHOLD_CHARS: usize = 12_000;
/// The summary circuit breaker's consecutive-failure threshold (Appendix A).
const DEFAULT_BREAKER_FAILURE_THRESHOLD: u32 = 2;
/// The summary circuit breaker's cooldown in seconds (Appendix A).
const DEFAULT_BREAKER_COOLDOWN_SECONDS: u64 = 300;
/// The expansion context budget in tokens (Appendix A).
const DEFAULT_EXPANSION_CONTEXT_TOKENS: usize = 32_000;
/// The per-expansion auxiliary-provider timeout in milliseconds (Appendix A).
const DEFAULT_EXPANSION_TIMEOUT_MS: u64 = 120_000;
/// The DAG depth retained across a `/new` session reset (`LCM:config.py:318`).
const DEFAULT_NEW_SESSION_RETAIN_DEPTH: i64 = 2;
/// The lifecycle row count above which the empty-row GC fires (`LCM:config.py:330`).
const DEFAULT_EMPTY_LIFECYCLE_GC_THRESHOLD: i64 = 200;
/// The empty-row GC age guard in hours (`LCM:config.py:335`).
const DEFAULT_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS: f64 = 24.0;
/// Hours between startup deep FTS integrity-checks (`_integrity_check_interval_hours`,
/// `LCM:db_bootstrap.py:331-348`).
const DEFAULT_FTS_INTEGRITY_CHECK_INTERVAL_HOURS: f64 = 24.0;
/// The sub-directory (under the data root) for externalized large payloads (§9.1).
const EXTERNALIZATION_SUBDIR: &str = "lcm-large-outputs";
/// The sub-directory (under the data root) for pre-compaction extraction markdown (§9.2).
const EXTRACTION_SUBDIR: &str = "lcm-extractions";

/// The default sensitive-pattern catalog names (§8.1).
fn default_sensitive_patterns() -> Vec<String> {
    vec![
        "api_key".to_string(),
        "bearer_token".to_string(),
        "password_assignment".to_string(),
        "private_key".to_string(),
    ]
}

/// The LCM context-engine configuration.
#[derive(Clone, Debug)]
pub struct LcmConfig {
    /// The data root for the store (default `$LCM_DATA_DIR` or `$HERMES_HOME/lcm/data`, else
    /// `./lcm-data`). Empty => in-memory.
    pub data_dir: PathBuf,
    /// The store name (`default` -> `{data_dir}/default.db`).
    pub bank: String,
    /// The fraction of the model context window at which compaction triggers (default `0.35`).
    pub context_threshold: f64,
    /// The number of most-recent turns always kept verbatim (default `32`).
    pub fresh_tail_count: usize,
    /// The base leaf-chunk size in tokens (default `20000`).
    pub leaf_chunk_tokens: usize,
    /// The sibling count that triggers condensation to the next depth (default `4`).
    pub condensation_fanin: usize,
    /// The maximum condensation depth — `0` disables, `-1` unlimited (default `3`).
    pub incremental_max_depth: i64,
    /// Let leaf compaction grow its working chunk with backlog pressure — the doubling ladder up
    /// to [`dynamic_leaf_chunk_max`](Self::dynamic_leaf_chunk_max) — and take up to 4 leaf passes
    /// per compaction (default `false`, `LCM:config.py:218`).
    pub dynamic_leaf_chunk_enabled: bool,
    /// The ceiling of the dynamic working leaf-chunk threshold in tokens (default `40000`,
    /// `LCM:config.py:220`).
    pub dynamic_leaf_chunk_max: usize,
    /// Suppress follow-on condensation right after a leaf pass unless enough same-depth debt has
    /// accumulated (default `false`, `LCM:config.py:223`) — keeps the assembled prefix
    /// cache-stable across consecutive turns.
    pub cache_friendly_condensation_enabled: bool,
    /// The minimum number of same-depth fanin groups before one cache-friendly follow-on
    /// condensation pass is allowed (default `2`, `LCM:config.py:226`).
    pub cache_friendly_min_debt_groups: usize,
    /// Persist raw-backlog maintenance debt in the lifecycle row and run bounded catch-up passes
    /// on later turns (default `false`, `LCM:config.py:229`).
    pub deferred_maintenance_enabled: bool,
    /// The maximum leaf passes a debt-triggered catch-up turn may spend (default `4`,
    /// `LCM:config.py:232`).
    pub deferred_maintenance_max_passes: usize,
    /// Bypass the cache-friendly/deferred polite gates once prompt pressure reaches this fraction
    /// of the context window (default `0.0` = disabled, `LCM:config.py:235`).
    pub critical_budget_pressure_ratio: f64,
    /// The L2 budget as a fraction of the L1 budget (default `0.50`).
    pub l2_budget_ratio: f64,
    /// The deterministic L3 truncation budget in tokens (default `512`).
    pub l3_truncate_tokens: usize,
    /// Hard cap for the assembled active context in tokens (default `0` = disabled,
    /// `LCM:config.py:245`).
    pub max_assembly_tokens: usize,
    /// Tokens reserved from the model context window before assembly — the effective cap becomes
    /// `context_length - reserve_tokens_floor` (default `0` = disabled, `LCM:config.py:248`).
    pub reserve_tokens_floor: usize,
    /// The per-summary auxiliary-provider timeout in milliseconds (default `60000`).
    pub summary_timeout_ms: u64,
    /// Custom instructions injected into every summarization prompt (default empty,
    /// `LCM:config.py:264`).
    pub custom_instructions: String,

    // ---- M5: ingest protection (§8/§9) -----------------------------------------------------
    /// Enable sensitive-pattern redaction at the ingest boundary (default `false`, §8.1).
    pub sensitive_patterns_enabled: bool,
    /// The active sensitive-pattern catalog names (default the four-name catalog, §8.1).
    pub sensitive_patterns: Vec<String>,
    /// Provenance of [`sensitive_patterns`](Self::sensitive_patterns) for `lcm_status`
    /// (`default` | `config`; Python's `default` | `env`, `LCM:config.py:283`).
    pub sensitive_patterns_source: String,
    /// Enable threshold externalization of oversized non-base64 payloads (default `false`, §9.1).
    /// The base64/data-URI storage guard (§8.2) is always on independently of this flag.
    pub large_output_externalization_enabled: bool,
    /// The character threshold for opt-in payload externalization (default `12000`, §9.1).
    pub large_output_externalization_threshold_chars: usize,
    /// Override directory for externalized payloads (default empty → `<data_root>/lcm-large-outputs`).
    pub large_output_externalization_path: String,
    /// Enable rewriting summarized+externalized tool rows to a placeholder after compaction
    /// (default `false`, §9.1 transcript GC).
    pub large_output_transcript_gc_enabled: bool,
    /// Enable pre-compaction extraction of decisions/commitments to a daily markdown (default
    /// `false`, §9.2).
    pub extraction_enabled: bool,
    /// The extraction aux-model selector (default empty → the summary/aux provider, §9.2). Not
    /// consulted by the engine itself — extraction routes through the primary aux route (§7.4
    /// deviation); a composition layer may use it to resolve that route (§12.4).
    pub extraction_model: String,
    /// Override directory for extraction markdown (default empty → `<data_root>/lcm-extractions`).
    pub extraction_output_path: String,

    // ---- M7: filters, routing, presets (§12.3-12.6) ----------------------------------------
    /// Session globs whose sessions are fully ignored (no ingest/compaction writes) (§12.5).
    pub ignore_session_patterns: Vec<String>,
    /// Provenance of the ignore-session list for `lcm_status` (`LCM:config.py:258`).
    pub ignore_session_patterns_source: String,
    /// Session globs whose sessions are read-only/stateless (no writes) (§12.5).
    pub stateless_session_patterns: Vec<String>,
    /// Provenance of the stateless-session list for `lcm_status` (`LCM:config.py:259`).
    pub stateless_session_patterns_source: String,
    /// Message-content regexes whose matching turns are filtered before the store (§12.3).
    pub ignore_message_patterns: Vec<String>,
    /// Provenance of the ignore-message list for `lcm_status` (`LCM:config.py:260`).
    pub ignore_message_patterns_source: String,
    /// The primary summary aux-model selector (default empty → the injected aux provider, §12.4).
    /// The engine reads it only for `lcm_status` display; routing is the injected aux chain.
    pub summary_model: String,
    /// The ordered summary fallback aux-model selectors (default empty, §7.3/§12.4). The engine
    /// applies per-route circuit breakers to whatever `aux_chain` the composition layer injects;
    /// this list is the config seam a composition layer resolves that chain from.
    pub summary_fallback_models: Vec<String>,
    /// The summary circuit breaker's failure threshold (default `2`, §7.3).
    pub summary_circuit_breaker_failure_threshold: u32,
    /// The summary circuit breaker's cooldown in seconds (default `300`, §7.3).
    pub summary_circuit_breaker_cooldown_seconds: u64,
    /// The expansion aux-model selector (default empty → summary/aux, §12.4).
    pub expansion_model: String,
    /// The expansion context budget in tokens (default `32000`, §10.5).
    pub expansion_context_tokens: usize,
    /// The per-expansion aux-provider timeout in milliseconds (default `120000`, §10.5).
    pub expansion_timeout_ms: u64,

    // ---- Session lifecycle (§13) -------------------------------------------------------------
    /// DAG depth retained across a `/new`-style session reset (default `2`,
    /// `LCM:config.py:318`): `-1` keeps all nodes, `0` deletes everything, `N` keeps depth >= N.
    pub new_session_retain_depth: i64,
    /// Safety gate for the destructive `/lcm doctor clean apply` operator workflow (default
    /// `false`, `LCM:config.py:320`; consumed by the `/lcm` command surface).
    pub doctor_clean_apply_enabled: bool,
    /// Enable pruning lifecycle rows for sessions that never ingested data (default `true`,
    /// `LCM:config.py:327`). Runs at session bind when the table exceeds the threshold.
    pub empty_lifecycle_gc_enabled: bool,
    /// The lifecycle row count above which the empty-row GC fires (default `200`,
    /// `LCM:config.py:330`).
    pub empty_lifecycle_gc_threshold: i64,
    /// Age guard for the empty-row GC — rows younger than this are kept because another live
    /// engine may not have ingested its first message yet (default `24.0` hours; `None` prunes
    /// regardless of age, `LCM:config.py:335`).
    pub empty_lifecycle_gc_max_age_hours: Option<f64>,

    // ---- Store hygiene (§4.4) -----------------------------------------------------------------
    /// Hours between startup deep FTS integrity-checks (default `24.0`; `0` checks on every open,
    /// negative never deep-checks on open — `_integrity_check_interval_hours`,
    /// `LCM:db_bootstrap.py:331-348`).
    pub fts_integrity_check_interval_hours: f64,
}

impl Default for LcmConfig {
    fn default() -> Self {
        // Pure data — no environment reads. The node binary injects `data_dir` (the profile home)
        // at construction; the compaction knobs carry the Appendix-A defaults.
        Self {
            data_dir: PathBuf::from("lcm-data"),
            bank: "default".to_string(),
            context_threshold: DEFAULT_CONTEXT_THRESHOLD,
            fresh_tail_count: DEFAULT_FRESH_TAIL_COUNT,
            leaf_chunk_tokens: DEFAULT_LEAF_CHUNK_TOKENS,
            condensation_fanin: DEFAULT_CONDENSATION_FANIN,
            incremental_max_depth: DEFAULT_INCREMENTAL_MAX_DEPTH,
            dynamic_leaf_chunk_enabled: false,
            dynamic_leaf_chunk_max: DEFAULT_DYNAMIC_LEAF_CHUNK_MAX,
            cache_friendly_condensation_enabled: false,
            cache_friendly_min_debt_groups: DEFAULT_CACHE_FRIENDLY_MIN_DEBT_GROUPS,
            deferred_maintenance_enabled: false,
            deferred_maintenance_max_passes: DEFAULT_DEFERRED_MAINTENANCE_MAX_PASSES,
            critical_budget_pressure_ratio: 0.0,
            l2_budget_ratio: DEFAULT_L2_BUDGET_RATIO,
            l3_truncate_tokens: DEFAULT_L3_TRUNCATE_TOKENS,
            max_assembly_tokens: 0,
            reserve_tokens_floor: 0,
            summary_timeout_ms: DEFAULT_SUMMARY_TIMEOUT_MS,
            custom_instructions: String::new(),

            sensitive_patterns_enabled: false,
            sensitive_patterns: default_sensitive_patterns(),
            sensitive_patterns_source: "default".to_string(),
            large_output_externalization_enabled: false,
            large_output_externalization_threshold_chars: DEFAULT_EXTERNALIZATION_THRESHOLD_CHARS,
            large_output_externalization_path: String::new(),
            large_output_transcript_gc_enabled: false,
            extraction_enabled: false,
            extraction_model: String::new(),
            extraction_output_path: String::new(),

            ignore_session_patterns: Vec::new(),
            ignore_session_patterns_source: "default".to_string(),
            stateless_session_patterns: Vec::new(),
            stateless_session_patterns_source: "default".to_string(),
            ignore_message_patterns: Vec::new(),
            ignore_message_patterns_source: "default".to_string(),
            summary_model: String::new(),
            summary_fallback_models: Vec::new(),
            summary_circuit_breaker_failure_threshold: DEFAULT_BREAKER_FAILURE_THRESHOLD,
            summary_circuit_breaker_cooldown_seconds: DEFAULT_BREAKER_COOLDOWN_SECONDS,
            expansion_model: String::new(),
            expansion_context_tokens: DEFAULT_EXPANSION_CONTEXT_TOKENS,
            expansion_timeout_ms: DEFAULT_EXPANSION_TIMEOUT_MS,

            new_session_retain_depth: DEFAULT_NEW_SESSION_RETAIN_DEPTH,
            doctor_clean_apply_enabled: false,
            empty_lifecycle_gc_enabled: true,
            empty_lifecycle_gc_threshold: DEFAULT_EMPTY_LIFECYCLE_GC_THRESHOLD,
            empty_lifecycle_gc_max_age_hours: Some(DEFAULT_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS),

            fts_integrity_check_interval_hours: DEFAULT_FTS_INTEGRITY_CHECK_INTERVAL_HOURS,
        }
    }
}

impl LcmConfig {
    /// An in-memory configuration (the SQLite store opens `:memory:`), for tests and ephemeral nodes.
    pub fn in_memory() -> Self {
        Self {
            data_dir: PathBuf::new(),
            bank: String::new(),
            ..Self::default()
        }
    }

    /// The resolved database path for the configured bank (empty `data_dir` => in-memory).
    pub fn db_path(&self) -> Option<PathBuf> {
        if self.data_dir.as_os_str().is_empty() {
            return None;
        }
        let name = if self.bank.is_empty() {
            "default"
        } else {
            &self.bank
        };
        Some(self.data_dir.join(format!("{name}.db")))
    }

    /// The directory for externalized large payloads (§9.1): the explicit override if set, else
    /// `<data_root>/lcm-large-outputs`, else `None` for an in-memory/ephemeral bank (the storage
    /// guard then leaves content inline rather than writing to disk).
    pub fn externalization_dir(&self) -> Option<PathBuf> {
        if !self.large_output_externalization_path.is_empty() {
            return Some(PathBuf::from(&self.large_output_externalization_path));
        }
        if self.data_dir.as_os_str().is_empty() {
            return None;
        }
        Some(self.data_dir.join(EXTERNALIZATION_SUBDIR))
    }

    /// The directory for pre-compaction extraction markdown (§9.2): the explicit override if set,
    /// else `<data_root>/lcm-extractions`, else `None` for an in-memory/ephemeral bank.
    pub fn extraction_dir(&self) -> Option<PathBuf> {
        if !self.extraction_output_path.is_empty() {
            return Some(PathBuf::from(&self.extraction_output_path));
        }
        if self.data_dir.as_os_str().is_empty() {
            return None;
        }
        Some(self.data_dir.join(EXTRACTION_SUBDIR))
    }

    /// The directory `/lcm backup` and `/lcm rotate apply` snapshots land in
    /// (`Engine.backup_dir`, `LCM:engine.py:4430-4442`, with the db parent standing in for the
    /// Hermes home): `<db parent>/backups/lcm`, `None` for an in-memory bank.
    pub fn backup_dir(&self) -> Option<PathBuf> {
        let db = self.db_path()?;
        let parent = db.parent().unwrap_or_else(|| Path::new("."));
        Some(parent.join("backups").join("lcm"))
    }

    /// The rolling rotate-latest snapshot slot (`Engine.rotate_backup_path`,
    /// `LCM:engine.py:4444-4451`): `<backup_dir>/<db stem>-rotate-latest.sqlite3`.
    pub fn rotate_backup_path(&self) -> Option<PathBuf> {
        let db = self.db_path()?;
        let stem = db.file_stem()?.to_string_lossy().into_owned();
        Some(
            self.backup_dir()?
                .join(format!("{stem}-rotate-latest.sqlite3")),
        )
    }

    /// The active assembly cap, if any (`_effective_assembly_token_cap`,
    /// `LCM:engine.py:4263-4289`): the tighter of the explicit `max_assembly_tokens` hard cap and
    /// the `context_length - reserve_tokens_floor` headroom reserve (each only when configured
    /// `> 0`; a reserve at or above the window disables the reserve-based cap with a warning).
    pub fn effective_assembly_token_cap(&self, context_length: Option<usize>) -> Option<usize> {
        let mut caps: Vec<usize> = Vec::new();
        if self.max_assembly_tokens > 0 {
            caps.push(self.max_assembly_tokens);
        }
        if let Some(window) = context_length.filter(|w| *w > 0) {
            if self.reserve_tokens_floor > 0 {
                if window > self.reserve_tokens_floor {
                    caps.push(window - self.reserve_tokens_floor);
                } else {
                    tracing::warn!(
                        reserve_tokens_floor = self.reserve_tokens_floor,
                        context_length = window,
                        "lcm: reserve_tokens_floor disables the reserve-based assembly cap \
                         because it is not below context_length"
                    );
                }
            }
        }
        caps.into_iter().min().map(|c| c.max(1))
    }
}
