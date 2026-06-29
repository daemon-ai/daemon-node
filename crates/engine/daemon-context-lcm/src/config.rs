// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Configuration for the LCM context engine (`daemon-context-lcm-port-spec.md` §12.1).
//!
//! The M1-M4 compaction fields plus the M5 (protection §8/§9) and M7 (filters/routing/presets
//! §12.3-12.6) knobs. Constants carry the Python defaults verbatim (Appendix A). Every M5/M7 opt-in
//! defaults *off* (empty pattern lists / disabled flags), so the zero-config default path is the
//! always-on compaction path only. The remaining escape hatches (dynamic leaf chunking,
//! deferred-maintenance debt, critical-pressure) stay unexposed — the default config never triggers
//! them.

use std::path::PathBuf;

/// Environment override for the LCM data directory.
const DATA_DIR_ENV: &str = "LCM_DATA_DIR";
/// Environment override for the compaction threshold (a fraction of the model context window).
const CONTEXT_THRESHOLD_ENV: &str = "LCM_CONTEXT_THRESHOLD";
/// Environment override for the verbatim-kept fresh-tail turn count.
const FRESH_TAIL_ENV: &str = "LCM_FRESH_TAIL_COUNT";

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
    /// The L2 budget as a fraction of the L1 budget (default `0.50`).
    pub l2_budget_ratio: f64,
    /// The deterministic L3 truncation budget in tokens (default `512`).
    pub l3_truncate_tokens: usize,
    /// The per-summary auxiliary-provider timeout in milliseconds (default `60000`).
    pub summary_timeout_ms: u64,

    // ---- M5: ingest protection (§8/§9) -----------------------------------------------------
    /// Enable sensitive-pattern redaction at the ingest boundary (default `false`, §8.1).
    pub sensitive_patterns_enabled: bool,
    /// The active sensitive-pattern catalog names (default the four-name catalog, §8.1).
    pub sensitive_patterns: Vec<String>,
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
    /// The extraction aux-model selector (default empty → the summary/aux provider, §9.2).
    pub extraction_model: String,
    /// Override directory for extraction markdown (default empty → `<data_root>/lcm-extractions`).
    pub extraction_output_path: String,

    // ---- M7: filters, routing, presets (§12.3-12.6) ----------------------------------------
    /// Session globs whose sessions are fully ignored (no ingest/compaction writes) (§12.5).
    pub ignore_session_patterns: Vec<String>,
    /// Session globs whose sessions are read-only/stateless (no writes) (§12.5).
    pub stateless_session_patterns: Vec<String>,
    /// Message-content regexes whose matching turns are filtered before the store (§12.3).
    pub ignore_message_patterns: Vec<String>,
    /// The primary summary aux-model selector (default empty → the injected aux provider, §12.4).
    pub summary_model: String,
    /// The ordered summary fallback aux-model selectors (default empty, §7.3/§12.4).
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
}

impl Default for LcmConfig {
    fn default() -> Self {
        let data_dir = std::env::var_os(DATA_DIR_ENV)
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HERMES_HOME").map(|h| PathBuf::from(h).join("lcm/data")))
            .unwrap_or_else(|| PathBuf::from("lcm-data"));
        let context_threshold = std::env::var(CONTEXT_THRESHOLD_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| *v > 0.0 && *v <= 1.0)
            .unwrap_or(DEFAULT_CONTEXT_THRESHOLD);
        let fresh_tail_count = std::env::var(FRESH_TAIL_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_FRESH_TAIL_COUNT);
        Self {
            data_dir,
            bank: "default".to_string(),
            context_threshold,
            fresh_tail_count,
            leaf_chunk_tokens: DEFAULT_LEAF_CHUNK_TOKENS,
            condensation_fanin: DEFAULT_CONDENSATION_FANIN,
            incremental_max_depth: DEFAULT_INCREMENTAL_MAX_DEPTH,
            l2_budget_ratio: DEFAULT_L2_BUDGET_RATIO,
            l3_truncate_tokens: DEFAULT_L3_TRUNCATE_TOKENS,
            summary_timeout_ms: DEFAULT_SUMMARY_TIMEOUT_MS,

            sensitive_patterns_enabled: false,
            sensitive_patterns: default_sensitive_patterns(),
            large_output_externalization_enabled: false,
            large_output_externalization_threshold_chars: DEFAULT_EXTERNALIZATION_THRESHOLD_CHARS,
            large_output_externalization_path: String::new(),
            large_output_transcript_gc_enabled: false,
            extraction_enabled: false,
            extraction_model: String::new(),
            extraction_output_path: String::new(),

            ignore_session_patterns: Vec::new(),
            stateless_session_patterns: Vec::new(),
            ignore_message_patterns: Vec::new(),
            summary_model: String::new(),
            summary_fallback_models: Vec::new(),
            summary_circuit_breaker_failure_threshold: DEFAULT_BREAKER_FAILURE_THRESHOLD,
            summary_circuit_breaker_cooldown_seconds: DEFAULT_BREAKER_COOLDOWN_SECONDS,
            expansion_model: String::new(),
            expansion_context_tokens: DEFAULT_EXPANSION_CONTEXT_TOKENS,
            expansion_timeout_ms: DEFAULT_EXPANSION_TIMEOUT_MS,
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
}
