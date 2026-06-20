//! [`Config`] — engine tunables (§20), loaded by the host and injected at construction.
//!
//! These are *engine* knobs, distinct from the host/node configuration (partition, socket, store
//! backend) that lives at the composition layer. The host reads them (from TOML/env) and hands them
//! to the engine via [`EngineProfile::with_config`](crate::EngineProfile::with_config); the turn
//! loop never reads env/files itself.

use serde::{Deserialize, Serialize};

/// The default per-turn ReAct iteration cap (model rounds), mirroring hermes' `iteration_budget.py`
/// parent default (90). It is the hard stop that terminates a turn's model<->tool loop.
pub const DEFAULT_MAX_ITERATIONS: u32 = 90;

/// The default per-tool result-byte budget: a tool result longer than this is truncated by the §12
/// pipeline so one tool cannot dominate the model context.
pub const DEFAULT_TOOL_RESULT_BUDGET: usize = 64 * 1024;

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
    /// The per-turn ReAct iteration cap (§20 iteration budget): the maximum number of model rounds in
    /// one turn's model->tools->model loop. On exhaustion the engine makes one final toolless summary
    /// call and ends the turn `BudgetExhausted`.
    pub max_iterations: u32,
    /// The per-tool result-byte budget (§12 sanitize+budget); `0` disables truncation.
    pub tool_result_budget: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_retry_attempts: 1,
            context_budget_tokens: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tool_result_budget: DEFAULT_TOOL_RESULT_BUDGET,
        }
    }
}
