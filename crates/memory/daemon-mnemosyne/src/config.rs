//! Engine configuration (data root, bank, session, decay/TTL knobs).
//!
//! Mirrors the Mnemosyne env-var surface (`MNEMOSYNE_DATA_DIR`, `MNEMOSYNE_RECENCY_HALFLIFE`,
//! `MNEMOSYNE_WM_TTL_HOURS`, ...). Current slice includes the fields referenced by the Rust port.

use std::path::PathBuf;

/// Which recall pipeline `Engine::recall` dispatches to (`beam.py` `recall` L5098 polyphonic reroute
/// / `recall_enhanced` L6202 gate). Selected from the Mnemosyne env flags, defaulting to `Base`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
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
    /// Working-memory TTL in hours (`WORKING_MEMORY_TTL_HOURS`, default 168).
    pub working_memory_ttl_hours: f64,
    /// Which recall pipeline to use (default [`RecallMode::Base`]).
    pub recall_mode: RecallMode,
    /// Enable the opt-in tier-2 LLM conflict detector in sleep (`MNEMOSYNE_LLM_CONFLICT_DETECTION`).
    pub llm_conflict_detection: bool,
}

impl Default for MnemosyneConfig {
    fn default() -> Self {
        let data_dir = std::env::var("MNEMOSYNE_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HERMES_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| std::env::temp_dir().join("hermes"));
                home.join("mnemosyne").join("data")
            });
        Self {
            data_dir,
            bank: "default".to_string(),
            session_id: "default".to_string(),
            recency_halflife_hours: 168.0,
            working_memory_ttl_hours: 168.0,
            recall_mode: recall_mode_from_env(),
            llm_conflict_detection: env_flag("MNEMOSYNE_LLM_CONFLICT_DETECTION"),
        }
    }
}

/// True if the named env var is set to `1` (the Mnemosyne flag convention).
fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

/// Select the recall pipeline from the Mnemosyne env flags. Polyphonic takes precedence over
/// enhanced (matching `recall`'s reroute order), both default off.
fn recall_mode_from_env() -> RecallMode {
    if env_flag("MNEMOSYNE_POLYPHONIC_RECALL") {
        RecallMode::Polyphonic
    } else if env_flag("MNEMOSYNE_ENHANCED_RECALL") {
        RecallMode::Enhanced
    } else {
        RecallMode::Base
    }
}

impl MnemosyneConfig {
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
}
