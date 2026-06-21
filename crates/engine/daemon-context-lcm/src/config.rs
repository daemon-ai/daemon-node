//! Configuration for the LCM context engine (`daemon-context-lcm-port-spec.md` §12.1).
//!
//! Only the fields the M1-M4 compaction path consumes are surfaced; the protection (§8), routing,
//! and preset knobs (§12.3-12.6) arrive with their milestones. Constants carry the Python defaults
//! verbatim (Appendix A). The opt-in escape hatches (dynamic leaf chunking, deferred-maintenance
//! debt, critical-pressure) are intentionally *not* exposed yet — the default config never triggers
//! them, so M1-M4 implements the always-on path only.

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
}
