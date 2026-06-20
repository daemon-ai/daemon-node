//! [`Config`] — engine tunables (§20), loaded by the host and injected at construction.
//!
//! These are *engine* knobs, distinct from the host/node configuration (partition, socket, store
//! backend) that lives at the composition layer. The host reads them (from TOML/env) and hands them
//! to the engine via [`EngineProfile::with_config`](crate::EngineProfile::with_config); the turn
//! loop never reads env/files itself. Phase 9 ships a small enforced subset (`model_retry_attempts`)
//! plus a defined-but-not-yet-enforced hint (`context_budget_tokens`); more land with later slices.

use serde::{Deserialize, Serialize};

/// Engine tunables governing one engine's turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// How many times `call_model` retries on a *rotatable* provider failure (quota/auth) before
    /// giving up. `1` reproduces the prior hardcoded single-retry behaviour.
    pub model_retry_attempts: u8,
    /// A soft context-token budget hint for `build_context` (not yet enforced; reserved for the
    /// compaction slice).
    pub context_budget_tokens: Option<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_retry_attempts: 1,
            context_budget_tokens: None,
        }
    }
}
