//! Engine configuration (data root, bank, session, decay/TTL knobs).
//!
//! Mirrors the Mnemosyne env-var surface (`MNEMOSYNE_DATA_DIR`, `MNEMOSYNE_RECENCY_HALFLIFE`,
//! `MNEMOSYNE_WM_TTL_HOURS`, ...). Scaffold: only the fields the spec references are present.

use std::path::PathBuf;

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
        }
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
