//! Configuration for the LCM context engine (env-overridable, matching the daemon's config style).

use std::path::PathBuf;

/// Environment override for the LCM data directory.
const DATA_DIR_ENV: &str = "LCM_DATA_DIR";
/// Environment override for the compaction threshold (a fraction of the model context window).
const THRESHOLD_PERCENT_ENV: &str = "LCM_THRESHOLD_PERCENT";

/// The LCM context-engine configuration.
#[derive(Clone, Debug)]
pub struct LcmConfig {
    /// The data root for the summary store (default `$LCM_DATA_DIR` or `$HERMES_HOME/lcm/data`,
    /// else `./lcm-data`).
    pub data_dir: PathBuf,
    /// The store name (`default` -> `{data_dir}/lcm.db`).
    pub bank: String,
    /// The fraction of the model context window at which compaction triggers (default `0.75`). Used
    /// to derive a threshold from [`ModelInfo::max_context`](daemon_core::context::ModelInfo) when the
    /// host does not set an explicit `context_budget_tokens`.
    pub threshold_percent: f64,
}

impl Default for LcmConfig {
    fn default() -> Self {
        let data_dir = std::env::var_os(DATA_DIR_ENV)
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HERMES_HOME").map(|h| PathBuf::from(h).join("lcm/data")))
            .unwrap_or_else(|| PathBuf::from("lcm-data"));
        let threshold_percent = std::env::var(THRESHOLD_PERCENT_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.75);
        Self {
            data_dir,
            bank: "default".to_string(),
            threshold_percent,
        }
    }
}

impl LcmConfig {
    /// An in-memory configuration (the SQLite store opens `:memory:`), for tests and ephemeral nodes.
    pub fn in_memory() -> Self {
        Self {
            data_dir: PathBuf::new(),
            bank: String::new(),
            threshold_percent: 0.75,
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
