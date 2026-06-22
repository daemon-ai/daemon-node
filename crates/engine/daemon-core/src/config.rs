//! [`Config`] ? engine tunables (?20), loaded by the host and injected at construction.
//!
//! These are *engine* knobs, distinct from the host/node configuration (partition, socket, store
//! backend) that lives at the composition layer. The host reads them (from TOML/env) and hands them
//! to the engine via [`EngineProfile::with_config`](crate::EngineProfile::with_config); the turn
//! loop never reads env/files itself.

use crate::approval::ApprovalPolicy;
use serde::{Deserialize, Serialize};

/// The default per-turn ReAct iteration cap (model rounds), mirroring hermes' `iteration_budget.py`
/// parent default (90). It is the hard stop that terminates a turn's model<->tool loop.
pub const DEFAULT_MAX_ITERATIONS: u32 = 90;

/// The default per-tool result-byte budget: a tool result longer than this is truncated by the ?12
/// pipeline so one tool cannot dominate the model context.
pub const DEFAULT_TOOL_RESULT_BUDGET: usize = 64 * 1024;

/// The default ?8 recovery budget: how many times a single `call_model` retries a *recoverable*
/// model failure (rate-limit/transport/overload/format) before giving up (hermes' parent default 3).
pub const DEFAULT_MODEL_MAX_RETRIES: u32 = 3;

/// The default ?8 backoff floor (ms): the base of the jittered exponential backoff (2s).
pub const DEFAULT_MODEL_BACKOFF_BASE_MS: u64 = 2_000;

/// The default ?8 backoff ceiling (ms): the cap on any single backoff sleep (120s).
pub const DEFAULT_MODEL_BACKOFF_MAX_MS: u64 = 120_000;

/// The default ?8 stale-stream watchdog (ms): a model stream that emits nothing for this long is
/// classified as a transient transport failure and recovered (180s).
pub const DEFAULT_MODEL_STREAM_WATCHDOG_MS: u64 = 180_000;

/// The default post-turn background-review nudge interval: `0` disables the engine-native trigger,
/// the conservative core default (the host opts in, e.g. hermes' `creation_nudge_interval` of 10).
/// When enabled, the engine emits an [`Effect::Spawn`](crate::turn::Effect::Spawn) for the matching
/// background-review `kind` and resets the counter (cf. hermes `turn_finalizer.py:375-401`).
pub const DEFAULT_REVIEW_NUDGE_INTERVAL: u32 = 0;

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
    /// The per-turn ReAct iteration cap (?20 iteration budget): the maximum number of model rounds in
    /// one turn's model->tools->model loop. On exhaustion the engine makes one final toolless summary
    /// call and ends the turn `BudgetExhausted`.
    pub max_iterations: u32,
    /// The per-tool result-byte budget (?12 sanitize+budget); `0` disables truncation.
    pub tool_result_budget: usize,
    /// The ?8 recovery budget: how many times one `call_model` retries a *recoverable* model
    /// failure (rate-limit/transport/overload/format) before giving up.
    pub model_max_retries: u32,
    /// The ?8 backoff floor (ms): the base of the jittered exponential backoff.
    pub model_backoff_base_ms: u64,
    /// The ?8 backoff ceiling (ms): the cap on any single backoff sleep.
    pub model_backoff_max_ms: u64,
    /// The ?8 stale-stream watchdog (ms): a model stream silent this long is recovered as a
    /// transient transport failure. `0` disables the watchdog.
    pub model_stream_watchdog_ms: u64,
    /// Post-turn **skill-review** nudge interval, in tool iterations accumulated since the last
    /// review or `skill_manage` use. When `> 0` and the counter reaches it, the engine emits an
    /// `Effect::Spawn { kind: "skill_review" }` and resets the counter. `0` disables (default).
    /// Mirrors hermes' `_skill_nudge_interval` / `skills.creation_nudge_interval`.
    pub skill_review_interval: u32,
    /// Post-turn **memory-review** nudge interval, in completed turns since the last review or
    /// memory write. When `> 0` and the counter reaches it, the engine emits an
    /// `Effect::Spawn { kind: "memory_review" }` and resets the counter. `0` disables (default).
    /// Mirrors hermes' `_memory_nudge_interval`.
    pub memory_review_interval: u32,
    /// The default edit-approval policy (?12 session mode) for a session that has not set an
    /// explicit one. The engine threads the effective policy onto each turn so a gated tool (fs
    /// edit / dangerous shell command) consults it. The default `Ask` makes the engine prompt for
    /// approval (the live host parks for a human; the durable host suspends for an operator);
    /// autonomous fleet engines are configured `AutoAllow` so they never stall.
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_retry_attempts: 1,
            context_budget_tokens: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tool_result_budget: DEFAULT_TOOL_RESULT_BUDGET,
            model_max_retries: DEFAULT_MODEL_MAX_RETRIES,
            model_backoff_base_ms: DEFAULT_MODEL_BACKOFF_BASE_MS,
            model_backoff_max_ms: DEFAULT_MODEL_BACKOFF_MAX_MS,
            model_stream_watchdog_ms: DEFAULT_MODEL_STREAM_WATCHDOG_MS,
            skill_review_interval: DEFAULT_REVIEW_NUDGE_INTERVAL,
            memory_review_interval: DEFAULT_REVIEW_NUDGE_INTERVAL,
            approval_policy: ApprovalPolicy::Ask,
        }
    }
}
