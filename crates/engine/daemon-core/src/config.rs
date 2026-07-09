// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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

/// The default no-progress guard: how many **consecutive identical** tool rounds (same calls + same
/// results) end the turn early (§4.2 loop guardrail). Complements [`DEFAULT_MAX_ITERATIONS`] — the
/// iteration cap bounds *total* rounds, this bounds a *stuck* model that keeps re-issuing the exact
/// same call and getting the exact same result without converging. `3` lets a model retry twice
/// before the engine concludes it is looping and ends `EndReason::NoProgress`.
pub const DEFAULT_MAX_REPEATED_ROUNDS: u32 = 3;

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

/// The default tool-search activation threshold (bytes of *deferrable* tool schema): when the summed
/// schema of the dynamic/long-tail tools (MCP + Python) exceeds this, the engine offers only the
/// `tool_search`/`tool_describe`/`tool_call` bridge instead of every deferrable schema, keeping the
/// prompt budget bounded (hermes' `tool_search` activates similarly). Core/built-in tools are always
/// offered. `0` disables collapsing (every tool is always offered). The default is generous enough
/// that a handful of dynamic tools stay inline; a large MCP fleet collapses behind search.
pub const DEFAULT_TOOL_SEARCH_THRESHOLD_BYTES: usize = 16 * 1024;

/// The default worker cap for a **parallel** tool batch: how many of a round's parallel-safe tool
/// calls run concurrently at once (the rest queue behind them). Mirrors hermes' `_MAX_TOOL_WORKERS`
/// (`run_agent.py`). Bounds resource use so a model emitting a huge parallel batch cannot spawn
/// unbounded concurrent work. Clamped to at least `1` at use.
pub const DEFAULT_MAX_PARALLEL_TOOLS: u32 = 8;

/// The default per-tool wall-clock timeout (ms) for the §12 pipeline timeout stage. `0` **disables**
/// the stage (the conservative default): a tool runs to completion unless it observes cancellation.
/// The host opts in with a positive value; self-limiting tools (`shell` foreground, `execute_code`)
/// override [`Tool::call_timeout`](crate::tools::Tool::call_timeout) to `None` to manage their own.
pub const DEFAULT_TOOL_TIMEOUT_MS: u64 = 0;

/// The default post-turn background-review nudge interval: `0` disables the engine-native trigger,
/// the conservative core default (the host opts in, e.g. hermes' `creation_nudge_interval` of 10).
/// When enabled, the engine emits an [`Effect::Spawn`](crate::turn::Effect::Spawn) for the matching
/// background-review `kind` and resets the counter (cf. hermes `turn_finalizer.py:375-401`).
pub const DEFAULT_REVIEW_NUDGE_INTERVAL: u32 = 0;

/// Whether the per-turn tool-call guardrail (§12) emits **warning** guidance by default. Warnings
/// never prevent a tool from running — they append a one-line nudge to the result so the model sees
/// the loop. Mirrors hermes `ToolCallGuardrailConfig.warnings_enabled` (default on).
pub const DEFAULT_GUARDRAIL_WARNINGS_ENABLED: bool = true;

/// Whether the per-turn tool-call guardrail (§12) may **hard-stop** (block a repeated call and end
/// the turn `NoProgress`) by default. Off by default (warn-only) so interactive sessions get a
/// gentle nudge; autonomous hosts opt in. Mirrors hermes `ToolCallGuardrailConfig.hard_stop_enabled`.
pub const DEFAULT_GUARDRAIL_HARD_STOP_ENABLED: bool = false;

/// Guardrail: warn after this many identical **failing** `(name, args)` calls in a turn (hermes
/// `exact_failure_warn_after`, 2).
pub const DEFAULT_EXACT_FAILURE_WARN_AFTER: u32 = 2;
/// Guardrail: block after this many identical **failing** `(name, args)` calls in a turn, when
/// hard-stop is enabled (hermes `exact_failure_block_after`, 5).
pub const DEFAULT_EXACT_FAILURE_BLOCK_AFTER: u32 = 5;
/// Guardrail: warn after this many **failures** of the same tool name (any args) in a turn (hermes
/// `same_tool_failure_warn_after`, 3).
pub const DEFAULT_SAME_TOOL_FAILURE_WARN_AFTER: u32 = 3;
/// Guardrail: halt after this many **failures** of the same tool name in a turn, when hard-stop is
/// enabled (hermes `same_tool_failure_halt_after`, 8).
pub const DEFAULT_SAME_TOOL_FAILURE_HALT_AFTER: u32 = 8;
/// Guardrail: warn after an **idempotent** (read-only) `(name, args)` call returns the same result
/// this many times in a turn (hermes `no_progress_warn_after`, 2).
pub const DEFAULT_NO_PROGRESS_WARN_AFTER: u32 = 2;
/// Guardrail: block after an **idempotent** `(name, args)` call returns the same result this many
/// times in a turn, when hard-stop is enabled (hermes `no_progress_block_after`, 5).
pub const DEFAULT_NO_PROGRESS_BLOCK_AFTER: u32 = 5;

/// Per-turn tool-call loop guardrail thresholds (§12), a port of hermes
/// [`ToolCallGuardrailConfig`](tool_guardrails.py). Distinct from the round-level no-progress guard
/// ([`Config::max_repeated_rounds`]): this tracks each `(name, args)` signature across the *whole*
/// turn (not just consecutive rounds) and escalates warn→block/halt with separate thresholds for
/// repeated-identical-failure, same-tool-failure, and idempotent-no-progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GuardrailConfig {
    /// Emit warning guidance on the result (never blocks execution).
    pub warnings_enabled: bool,
    /// Allow hard stops (block a call + end the turn `NoProgress`). Off ⇒ warn-only.
    pub hard_stop_enabled: bool,
    /// Warn after N identical failing `(name, args)` calls.
    pub exact_failure_warn_after: u32,
    /// Block after N identical failing `(name, args)` calls (hard-stop only).
    pub exact_failure_block_after: u32,
    /// Warn after N failures of the same tool name (any args).
    pub same_tool_failure_warn_after: u32,
    /// Halt after N failures of the same tool name (hard-stop only).
    pub same_tool_failure_halt_after: u32,
    /// Warn after an idempotent call returns the same result N times.
    pub no_progress_warn_after: u32,
    /// Block after an idempotent call returns the same result N times (hard-stop only).
    pub no_progress_block_after: u32,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            warnings_enabled: DEFAULT_GUARDRAIL_WARNINGS_ENABLED,
            hard_stop_enabled: DEFAULT_GUARDRAIL_HARD_STOP_ENABLED,
            exact_failure_warn_after: DEFAULT_EXACT_FAILURE_WARN_AFTER,
            exact_failure_block_after: DEFAULT_EXACT_FAILURE_BLOCK_AFTER,
            same_tool_failure_warn_after: DEFAULT_SAME_TOOL_FAILURE_WARN_AFTER,
            same_tool_failure_halt_after: DEFAULT_SAME_TOOL_FAILURE_HALT_AFTER,
            no_progress_warn_after: DEFAULT_NO_PROGRESS_WARN_AFTER,
            no_progress_block_after: DEFAULT_NO_PROGRESS_BLOCK_AFTER,
        }
    }
}

/// Engine tunables governing one engine's turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// How many times `call_model` retries on a *rotatable* provider failure (quota/auth) before
    /// giving up. `1` reproduces the prior hardcoded single-retry behaviour.
    pub model_retry_attempts: u8,
    /// The effective context-token budget. When set it is the target `prepare_turn_context` compacts
    /// the conversation to before a turn (the §10 pre-turn pressure check), and the C6 hard last-resort
    /// cap drops oldest turns to it if the context engine's own compaction leaves the conversation over
    /// budget. `None` => the engine's own threshold governs (a stateful engine like LCM sizes one from
    /// the model window in `on_model`); the budgeted default then never reports over budget.
    pub context_budget_tokens: Option<u32>,
    /// The per-turn ReAct iteration cap (?20 iteration budget): the maximum number of model rounds in
    /// one turn's model->tools->model loop. On exhaustion the engine makes one final toolless summary
    /// call and ends the turn `BudgetExhausted`.
    pub max_iterations: u32,
    /// The no-progress guard: how many **consecutive identical** tool rounds (same `(name,args)`
    /// calls producing the same results) end the turn early with `EndReason::NoProgress` — a stuck
    /// model that keeps repeating itself without converging, caught well before the `max_iterations`
    /// hard stop. `0` disables the guard.
    pub max_repeated_rounds: u32,
    /// The per-tool result-byte budget (?12 sanitize+budget); `0` disables truncation.
    pub tool_result_budget: usize,
    /// The worker cap for a **parallel** tool batch (§12): at most this many parallel-safe calls in
    /// one round run concurrently (via `.buffered`), the rest queue. Clamped to at least `1`.
    pub max_parallel_tools: u32,
    /// The per-tool wall-clock timeout (ms) for the §12 pipeline timeout stage. `0` disables it (the
    /// default); a self-limiting tool opts out via [`Tool::call_timeout`](crate::tools::Tool::call_timeout).
    pub tool_timeout_ms: u64,
    /// The per-turn tool-call loop guardrail thresholds (§12), a port of hermes `tool_guardrails.py`.
    pub guardrail: GuardrailConfig,
    /// The tool-search activation threshold (bytes of deferrable tool schema). When the summed
    /// deferrable schema exceeds this, the engine offers the `tool_search`/`tool_describe`/`tool_call`
    /// bridge in place of every deferrable schema. `0` disables collapsing (offer all tools always).
    pub tool_search_threshold_bytes: usize,
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
    /// The prompt-cache TTL threaded onto every request's breakpoints (§ prompt caching): the
    /// default 5-minute ephemeral tier, or the extended 1-hour tier for long-lived sessions
    /// (hermes `cache_ttl`). Parse operator strings via
    /// [`CacheTtl::from_config_str`](crate::provider::CacheTtl::from_config_str) — invalid values
    /// fall back to the default.
    #[serde(default)]
    pub cache_ttl: crate::provider::CacheTtl,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_retry_attempts: 1,
            context_budget_tokens: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_repeated_rounds: DEFAULT_MAX_REPEATED_ROUNDS,
            tool_result_budget: DEFAULT_TOOL_RESULT_BUDGET,
            max_parallel_tools: DEFAULT_MAX_PARALLEL_TOOLS,
            tool_timeout_ms: DEFAULT_TOOL_TIMEOUT_MS,
            guardrail: GuardrailConfig::default(), /* hermes tool_guardrails.py parity */
            tool_search_threshold_bytes: DEFAULT_TOOL_SEARCH_THRESHOLD_BYTES,
            model_max_retries: DEFAULT_MODEL_MAX_RETRIES,
            model_backoff_base_ms: DEFAULT_MODEL_BACKOFF_BASE_MS,
            model_backoff_max_ms: DEFAULT_MODEL_BACKOFF_MAX_MS,
            model_stream_watchdog_ms: DEFAULT_MODEL_STREAM_WATCHDOG_MS,
            skill_review_interval: DEFAULT_REVIEW_NUDGE_INTERVAL,
            memory_review_interval: DEFAULT_REVIEW_NUDGE_INTERVAL,
            approval_policy: ApprovalPolicy::Ask,
            cache_ttl: crate::provider::CacheTtl::FiveMin,
        }
    }
}
