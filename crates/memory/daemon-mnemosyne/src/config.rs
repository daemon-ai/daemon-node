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
    /// Multi-agent identity: the original writer (`MNEMOSYNE_AUTHOR_ID`, `beam.py` ctor L2616). When
    /// set it both stamps new rows and widens recall scope to all sessions for that author.
    pub author_id: Option<String>,
    /// Multi-agent identity: author type — `human`/`agent`/`system` (`MNEMOSYNE_AUTHOR_TYPE`).
    pub author_type: Option<String>,
    /// Multi-agent identity: channel/group id (`MNEMOSYNE_CHANNEL_ID`, `beam.py` ctor L2618). Writes
    /// default it to `session_id`; recall only filters on it when explicitly set.
    pub channel_id: Option<String>,
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

impl Default for MnemosyneConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("mnemosyne-data"),
            bank: "default".to_string(),
            session_id: "default".to_string(),
            recency_halflife_hours: 168.0,
            working_memory_ttl_hours: 168.0,
            recall_mode: RecallMode::Base,
            llm_conflict_detection: false,
            author_id: None,
            author_type: None,
            channel_id: None,
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
}
