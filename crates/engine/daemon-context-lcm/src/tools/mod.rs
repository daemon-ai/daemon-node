// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The seven `lcm_*` drill-down tools (§10).
//!
//! Ported from `LCM:tools.py`. Each handler takes parsed JSON args and returns a JSON string. They
//! are `ContextEngine`-owned (not in the main `ToolRegistry`): the host registers thin adapters that
//! resolve the calling session's [`LcmContextEngine`](crate::LcmContextEngine) and delegate to
//! [`LcmContextEngine::call_tool`], which builds a [`ToolCx`] and calls [`dispatch`].
//!
//! Scope divergence from the Python plugin (intentional, §14): tools read the **durable store** (the
//! full per-turn transcript ingested in `before_turn`) rather than the live `Conversation`, so a
//! `store_id`/`node_id` recovers exact content regardless of what is currently in-context.
//!
//! Module layout mirrors the Python tool groups: [`parse`] (arg coercion / content slicing),
//! [`grep`] (`lcm_grep` + `lcm_load_session`), [`expand`] (`lcm_describe` + `lcm_expand` + the
//! shared source-expansion helpers), [`query`] (`lcm_expand_query`), and [`diagnostics`]
//! (`lcm_status` + `lcm_doctor`).

pub mod schemas;

mod diagnostics;
mod expand;
mod grep;
mod parse;
mod query;

use crate::config::LcmConfig;
use crate::provider::UsageMetrics;
use crate::store::Store;
use crate::tokens::Tokenizer;
use daemon_core::tools::ToolDef;
use daemon_core::Provider;
use serde_json::{json, Value};

/// The stable names of the seven tools (the order `tool_defs` / `ContextEngine::tools` report).
pub const TOOL_NAMES: [&str; 7] = [
    "lcm_grep",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
    "lcm_doctor",
];

/// The §12 [`ToolDef`]s for all seven tools (session-independent — enumerate once).
pub fn tool_defs() -> Vec<ToolDef> {
    let schemas = [
        schemas::LCM_GREP,
        schemas::LCM_LOAD_SESSION,
        schemas::LCM_DESCRIBE,
        schemas::LCM_EXPAND,
        schemas::LCM_EXPAND_QUERY,
        schemas::LCM_STATUS,
        schemas::LCM_DOCTOR,
    ];
    TOOL_NAMES
        .iter()
        .zip(schemas)
        .map(|(name, schema)| ToolDef {
            name: (*name).to_string(),
            schema: schema.to_string(),
        })
        .collect()
}

/// Everything a tool handler needs, assembled per-call by [`LcmContextEngine::call_tool`].
pub(crate) struct ToolCx<'a> {
    /// The bank store (shared; row-scoped by session).
    pub store: &'a Store,
    /// The engine config (thresholds, fresh-tail count, paths).
    pub config: &'a LcmConfig,
    /// The model-aware token counter.
    pub tokenizer: &'a Tokenizer,
    /// The auxiliary provider (for `lcm_expand_query`).
    pub aux: &'a dyn Provider,
    /// The foreground session id (the §14.1 identity invariant).
    pub session_id: &'a str,
    /// The model-window-derived compaction threshold, if known (status/doctor).
    pub threshold_tokens: Option<usize>,
    /// The model context window in tokens, if known (drives the preset suggestion).
    pub context_length: Option<usize>,
    /// The token count of the most recent assembled prompt (`before_turn`) — backs the
    /// `context_pressure` doctor check.
    pub last_prompt_tokens: usize,
    /// How many compactions have run this incarnation (status).
    pub compaction_count: u64,
    /// Whether the session is ignored (no ingest/compaction) — §12.5.
    pub session_ignored: bool,
    /// Whether the session is stateless (read-only) — §12.5.
    pub session_stateless: bool,
    /// The process-lifetime ignored-message count (§12.5).
    pub ignored_message_count: u64,
    /// Provider-reported usage from the most recent response (`after_response`) — status.
    pub usage: UsageMetrics,
    /// The active model id (`on_model`) — status.
    pub model: &'a str,
    /// Where the context window came from (`model_info` / `default`) — status.
    pub context_length_source: &'a str,
    /// The most recent compaction outcome (`idle` / `compacted` / `noop`) — status.
    pub last_compression_status: &'a str,
    /// Why the most recent compaction was a no-op (empty when it wasn't) — status.
    pub last_compression_noop_reason: &'a str,
}

/// Dispatch one `lcm_*` tool by name, returning a JSON string (§10.7).
pub(crate) async fn dispatch(cx: &ToolCx<'_>, name: &str, args: Value) -> String {
    match name {
        "lcm_grep" => grep::grep(cx, &args),
        "lcm_load_session" => grep::load_session(cx, &args),
        "lcm_describe" => expand::describe(cx, &args),
        "lcm_expand" => expand::expand(cx, &args),
        "lcm_expand_query" => query::expand_query(cx, &args).await,
        "lcm_status" => diagnostics::status(cx),
        "lcm_doctor" => diagnostics::doctor(cx),
        other => json!({"status": "unknown_tool", "tool": other}).to_string(),
    }
}

/// The Python tool error shape: `{"error": message}` (`LCM:tools.py` returns bare error dicts).
pub(super) fn err(message: &str) -> String {
    json!({"error": message}).to_string()
}

#[cfg(test)]
mod tests;
