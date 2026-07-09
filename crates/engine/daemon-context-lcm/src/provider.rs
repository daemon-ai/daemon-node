// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`LcmContextEngine`] — the `daemon-core` [`ContextEngine`] (§10) backed by the summary DAG.
//!
//! `on_model` sizes the compaction threshold + selects the tokenizer from the model window;
//! `before_turn` measures token [`Pressure`] (with a boundary cooldown after a no-op compaction);
//! `compact` runs the real LCM pass (`compaction::run_compaction`) — summarize the region outside
//! the fresh tail into the DAG and reassemble `[system] + [summary] + [fresh tail]`; the
//! session-lifecycle hooks bind the conversation frontier.

use crate::compaction::{
    critical_budget_pressure_reached, leading_scaffold_count, leaf_compaction_candidate_status,
    refresh_raw_backlog_debt, run_compaction, should_run_deferred_maintenance,
};
use crate::config::LcmConfig;
use crate::error::Result;
use crate::escalation::SummaryCircuitBreaker;
use crate::ingest::flatten_turns;
use crate::patterns::{build_session_match_keys, MessagePatterns, SessionGlobs};
use crate::protection::{protect_message_for_ingest, sanitize_replay_turn, ReplayQuarantine};
use crate::store::Store;
use crate::tokens::Tokenizer;
use crate::tools::{ToolCx, TOOL_NAMES};
use async_trait::async_trait;
use daemon_common::SessionId;
use daemon_core::tools::ToolDef;
use daemon_core::{
    CommandCx, CommandError, CommandInvocation, CommandOutput, CommandProvider,
    CommandProviderHandle, CommandSpec, ContextEngine, Conversation, ModelInfo, Pressure, Provider,
    Turn,
};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Process-lifetime count of messages dropped by `ignore_message_patterns` across every session/
/// engine instance (§12.5); surfaced by `lcm_status`.
static IGNORED_MESSAGE_COUNT: AtomicU64 = AtomicU64::new(0);

/// The process-lifetime ignored-message count (`lcm_status`).
pub fn ignored_message_count() -> u64 {
    IGNORED_MESSAGE_COUNT.load(Ordering::Relaxed)
}

/// After a compaction that could not shrink anything (region already inside the fresh tail), suppress
/// re-triggering for this long so the engine doesn't re-attempt a no-op every turn (§6.3 cooldown).
const BOUNDARY_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(60);

/// Mutable per-session runtime state (small; behind a mutex so the sync hooks can update it).
#[derive(Default)]
struct State {
    /// The compaction threshold (model-window-derived), if known.
    threshold_tokens: Option<usize>,
    /// The model context window in tokens (drives the preset suggestion), if known.
    context_length: Option<usize>,
    /// The active model name (`on_model`) — `lcm_status`'s `model` field.
    model: String,
    /// Where `context_length` came from (`LCM:engine.py:_context_length_source`): `model_info`
    /// once `on_model` supplied a window, empty before that.
    context_length_source: String,
    /// The active session id (keys summary nodes / lifecycle rows).
    session_id: String,
    /// Whether the bound session matches an `ignore_session_patterns` glob (no ingest/compaction).
    session_ignored: bool,
    /// Whether the bound session matches a `stateless_session_patterns` glob (read-only; no writes).
    session_stateless: bool,
    /// The model-aware token counter (heuristic until `on_model`).
    tokenizer: Tokenizer,
    /// The per-route aux circuit breakers (one per `aux_chain` entry), carried across compactions.
    breakers: Vec<SummaryCircuitBreaker>,
    /// How many compactions have actually run this incarnation.
    compaction_count: u64,
    /// The token count of the most recent assembled prompt (measured in `before_turn`) — backs the
    /// `context_pressure` doctor check (the `engine.last_prompt_tokens` analog).
    last_prompt_tokens: usize,
    /// When the last compaction was a no-op (for the boundary cooldown).
    last_noop_at: Option<Instant>,
    /// The number of live conversation turns already ingested into `messages` this incarnation.
    cursor: usize,
    /// Per-turn ingest index: `turn_store_ids[i]` are the `store_id`s persisted for live turn `i`
    /// (empty for a synthetic summary turn). Kept aligned with the live conversation so compaction
    /// can attribute D0 `source_ids` without re-ingesting.
    turn_store_ids: Vec<Vec<i64>>,
    /// `lcm_status`'s `last_compression_status` (`LCM:engine.py:381`): `idle` until a compaction is
    /// attempted, then `compacted`/`noop` (empty means `idle`).
    last_compression_status: String,
    /// The reason of the last no-op compaction (`LCM:engine.py:382`); empty otherwise.
    last_compression_noop_reason: String,
    /// Whether this incarnation has reconciled its tail against the durable store yet (once per
    /// incarnation, on the first ingest).
    reconciled: bool,
    /// A pending reset boundary — the reset session + its frontier at reset time. The next
    /// *different*-session bind finalizes it; a same-session rebind clears it
    /// (`_pending_reset_*`, `LCM:engine.py:1558-1584/2202-2205`).
    pending_reset: Option<(String, i64)>,
    /// Provider-reported usage from the most recent model response (`update_from_response`,
    /// `LCM:engine.py:614-629`) — surfaced by `lcm_status`.
    usage: UsageMetrics,
    /// The last restart-reconciliation record (`_record_ingest_reconciliation`,
    /// `LCM:engine.py:3065-3085`) — surfaced by `lcm_status` as `ingest_reconciliation`.
    /// `Null` renders as the Python "not run" default. Counts are turn-based (adaptation).
    last_ingest_reconciliation: Value,
}

/// The provider-reported usage snapshot recorded by [`ContextEngine::after_response`]
/// (`update_from_response`, `LCM:engine.py:614-629`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UsageMetrics {
    /// Prompt/input tokens of the last response.
    pub last_input_tokens: u64,
    /// Completion/output tokens of the last response.
    pub last_output_tokens: u64,
    /// Prompt tokens served from the provider's cache.
    pub last_cache_read_tokens: u64,
    /// Prompt tokens written to the provider's cache.
    pub last_cache_write_tokens: u64,
    /// Reasoning/thinking tokens of the last response.
    pub last_reasoning_tokens: u64,
    /// Whether the provider surfaced cache metrics at all (`UsageDelta` carries no key-presence
    /// signal, so this is best-effort: true once either cache counter is nonzero).
    pub cache_metrics_available: bool,
}

impl UsageMetrics {
    /// `cache_read_ratio` (`LCM:engine.py:631-635`): cache-read tokens over prompt tokens.
    pub fn cache_read_ratio(&self) -> f64 {
        if self.last_input_tokens == 0 {
            return 0.0;
        }
        self.last_cache_read_tokens as f64 / self.last_input_tokens as f64
    }
}

/// The LCM context engine over a single summary-store bank.
pub struct LcmContextEngine {
    config: LcmConfig,
    store: Store,
    /// The primary aux provider (tools/extraction/expand_query) — `aux_chain[0]`.
    aux: Arc<dyn Provider>,
    /// The summarization fallback chain (`summary_model` then `summary_fallback_models`). The
    /// minimal port wires a single provider, so this has length 1 by default (§7.4 / §12.4).
    aux_chain: Vec<Arc<dyn Provider>>,
    /// Compiled `ignore_session_patterns` globs (§12.5).
    ignore_session_globs: SessionGlobs,
    /// Compiled `stateless_session_patterns` globs (§12.5).
    stateless_session_globs: SessionGlobs,
    /// Compiled `ignore_message_patterns` (§12.3).
    message_patterns: MessagePatterns,
    state: Mutex<State>,
}

impl LcmContextEngine {
    /// Open the engine for the configured bank (in-memory when `config.db_path()` is `None`), using
    /// `aux` as the auxiliary summarization provider.
    pub fn open(config: LcmConfig, aux: Arc<dyn Provider>) -> Result<Self> {
        let store = Store::open_at(
            config.db_path().as_deref(),
            config.fts_integrity_check_interval_hours,
        )?;
        let aux_chain = vec![aux.clone()];
        let breakers = aux_chain
            .iter()
            .map(|_| {
                SummaryCircuitBreaker::with_config(
                    config.summary_circuit_breaker_failure_threshold,
                    config.summary_circuit_breaker_cooldown_seconds,
                )
            })
            .collect();
        let ignore_session_globs = SessionGlobs::compile(&config.ignore_session_patterns);
        let stateless_session_globs = SessionGlobs::compile(&config.stateless_session_patterns);
        let message_patterns = MessagePatterns::compile(&config.ignore_message_patterns);
        Ok(Self {
            config,
            store,
            aux,
            aux_chain,
            ignore_session_globs,
            stateless_session_globs,
            message_patterns,
            state: Mutex::new(State {
                breakers,
                ..State::default()
            }),
        })
    }

    /// Recompute the session-filter flags (`session_ignored`/`session_stateless`) from the compiled
    /// globs for `session_id` (`_refresh_session_filters`, §12.5). The platform is unknown in
    /// daemon-core, so match keys reduce to the bare `session_id`.
    fn refresh_session_filters(&self, session_id: &str, state: &mut State) {
        let keys = build_session_match_keys("", session_id);
        state.session_ignored = self.ignore_session_globs.matches(&keys);
        state.session_stateless = self.stateless_session_globs.matches(&keys);
        if state.session_ignored || state.session_stateless {
            tracing::debug!(
                session = %session_id,
                ignored = state.session_ignored,
                stateless = state.session_stateless,
                "lcm: session filter active (no store writes)"
            );
        }
    }

    /// Open an in-memory engine (tests / ephemeral nodes) with the given aux provider.
    pub fn open_in_memory(aux: Arc<dyn Provider>) -> Result<Self> {
        Self::open(LcmConfig::in_memory(), aux)
    }

    /// Open an engine already bound to `session` — the per-session construction seam (the
    /// composition layer's [`ContextEngineBuilder`](daemon_core::ContextEngineBuilder)). Each session
    /// gets its own instance (so runtime `state` is never shared); all instances still share the
    /// bank's `lcm.db`, row-scoped by `session_id`.
    pub fn open_for_session(
        config: LcmConfig,
        session: &SessionId,
        aux: Arc<dyn Provider>,
    ) -> Result<Self> {
        let engine = Self::open(config, aux)?;
        {
            let mut state = engine.state.lock().expect("lcm state poisoned");
            state.session_id = session.as_str().to_string();
            state.tokenizer = Tokenizer::heuristic();
            engine.refresh_session_filters(session.as_str(), &mut state);
        }
        engine.bind_lifecycle(session.as_str());
        Ok(engine)
    }

    /// Bind `session_id`'s lifecycle row (`_bind_lifecycle_state`, `LCM:engine.py:1181-1218`), then
    /// garbage-collect empty lifecycle rows when the table exceeds the configured threshold —
    /// gateway restarts, ephemeral cron ticks, and crash-loops all create rows that never ingest
    /// data, so they are pruned here (age-guarded; the bound session is protected).
    fn bind_lifecycle(&self, session_id: &str) {
        let _ = self.store.bind_session(session_id, session_id, Self::now());
        if !self.config.empty_lifecycle_gc_enabled {
            return;
        }
        let over_threshold = self
            .store
            .lifecycle_row_count()
            .is_ok_and(|n| n > self.config.empty_lifecycle_gc_threshold);
        if !over_threshold {
            return;
        }
        let protected = vec![session_id.to_string()];
        match self.store.prune_empty_sessions(
            &protected,
            self.config.empty_lifecycle_gc_max_age_hours,
            Self::now(),
        ) {
            Ok(deleted) if deleted > 0 => tracing::info!(
                deleted,
                threshold = self.config.empty_lifecycle_gc_threshold,
                "lcm: pruned lifecycle rows with zero stored data"
            ),
            Ok(_) => {}
            Err(e) => tracing::debug!(error = %e, "lcm: empty-lifecycle GC failed"),
        }
    }

    /// Reset the per-session runtime state for a fresh/unproven session binding
    /// (`_reset_session_scoped_runtime_state`, `LCM:engine.py:1704-1742`): counters, compaction
    /// progress, and the ingest index. The next ingest reconciles against the durable frontier.
    fn reset_session_scoped_runtime_state(state: &mut State) {
        state.compaction_count = 0;
        state.last_prompt_tokens = 0;
        state.last_noop_at = None;
        state.cursor = 0;
        state.turn_store_ids.clear();
        state.reconciled = false;
        state.last_compression_status = "idle".to_string();
        state.last_compression_noop_reason.clear();
        // Back to the "not run" default (`LCM:engine.py:1732`).
        state.last_ingest_reconciliation = Value::Null;
    }

    /// The `/new`-style session reset (`on_session_reset`, `LCM:engine.py:2202-2219`): arm the
    /// reset boundary (finalized when the next session binds), stamp the lifecycle reset (clearing
    /// debt), reset the runtime state, and prune the bound session's retained DAG per
    /// `new_session_retain_depth` (`-1` keeps all, `0` deletes everything, `N` keeps depth >= N).
    /// The [`ContextEngine::on_session_reset`] trait hook delegates here (the engine fires it on a
    /// full-clear rewind), so a host no longer has to call this inherently.
    pub fn reset_bound_session(&self) {
        let session_id = {
            let state = self.state.lock().expect("lcm state poisoned");
            state.session_id.clone()
        };
        if session_id.is_empty() {
            return;
        }
        let frontier = self.store.get_frontier(&session_id).unwrap_or(0);
        {
            let mut state = self.state.lock().expect("lcm state poisoned");
            state.pending_reset = Some((session_id.clone(), frontier));
            Self::reset_session_scoped_runtime_state(&mut state);
        }
        let _ = self.store.record_reset(&session_id, Self::now());
        let retain = self.config.new_session_retain_depth;
        if retain == -1 {
            return;
        }
        let pruned = if retain == 0 {
            self.store.delete_session_nodes(&session_id)
        } else {
            self.store.delete_below_depth(&session_id, retain)
        };
        if let Ok(n) = pruned {
            if n > 0 {
                tracing::debug!(session = %session_id, deleted = n, retain, "lcm: reset pruned retained DAG");
            }
        }
    }

    /// Move retained summaries from `old_session_id` into `new_session_id`
    /// (`carry_over_new_session_context`, `LCM:engine.py:2221-2238`) — `/new` carry-over. Node ids
    /// and node-to-node lineage are preserved; descendant raw-message lineage is not rewritten.
    /// Returns the number of moved nodes.
    pub fn carry_over_new_session_context(
        &self,
        old_session_id: &str,
        new_session_id: &str,
    ) -> usize {
        if old_session_id.is_empty()
            || new_session_id.is_empty()
            || old_session_id == new_session_id
        {
            return 0;
        }
        {
            let state = self.state.lock().expect("lcm state poisoned");
            if state.session_ignored && new_session_id == state.session_id {
                tracing::debug!(session = %new_session_id, "lcm: carry-over skipped for ignored session");
                return 0;
            }
        }
        self.store
            .reassign_session_nodes(old_session_id, new_session_id)
            .unwrap_or(0)
    }

    /// Complete a `/new`-style rollover in one call (`rollover_session`,
    /// `LCM:engine.py:2240-2305`, the non-compression path): flush + finalize the old session,
    /// reset retained DAG state, bind the new session, and (optionally) carry retained summaries
    /// over. `conv` is the old session's final conversation for the last flush. Returns the number
    /// of carried-over nodes. The [`ContextEngine::rollover_session`] trait hook delegates here
    /// (dropping the count); this inherent variant keeps it for callers that report it.
    pub fn rollover_sessions(
        &self,
        old_session_id: &str,
        new_session_id: &str,
        conv: Option<&Conversation>,
        carry_over_context: bool,
    ) -> usize {
        let bound = {
            let state = self.state.lock().expect("lcm state poisoned");
            state.session_id.clone()
        };
        let can_carry_over = !old_session_id.is_empty() && old_session_id == bound;
        if can_carry_over {
            if let Some(conv) = conv {
                self.ingest_current(conv);
            }
            let frontier = self.store.get_frontier(old_session_id).unwrap_or(0);
            let _ =
                self.store
                    .finalize_session(old_session_id, old_session_id, frontier, Self::now());
            self.reset_bound_session();
        } else if !old_session_id.is_empty() {
            tracing::warn!(
                old = %old_session_id,
                bound = %bound,
                "lcm: rollover old session does not match bound session; skipping finalize/carry-over"
            );
        }
        self.on_session_start(&SessionId::new(new_session_id));
        if !carry_over_context || !can_carry_over {
            return 0;
        }
        self.carry_over_new_session_context(old_session_id, new_session_id)
    }

    /// The §12 [`ToolDef`]s for the seven `lcm_*` drill-down tools (session-independent — the host
    /// enumerates these once and registers an adapter per def that routes to [`Self::call_tool`]).
    pub fn tool_defs(&self) -> Vec<ToolDef> {
        crate::tools::tool_defs()
    }

    /// Dispatch one `lcm_*` tool by name against this engine's session/store, returning a JSON
    /// string (§10.7). The tools read the durable store, so recovery works regardless of what is
    /// currently in-context.
    pub async fn call_tool(&self, name: &str, args: Value) -> String {
        struct Snapshot {
            session_id: String,
            tokenizer: Tokenizer,
            threshold_tokens: Option<usize>,
            context_length: Option<usize>,
            last_prompt_tokens: usize,
            compaction_count: u64,
            session_ignored: bool,
            session_stateless: bool,
            usage: UsageMetrics,
            model: String,
            context_length_source: String,
            last_compression_status: String,
            last_compression_noop_reason: String,
            ingest_reconciliation: Value,
        }
        let snap = {
            let state = self.state.lock().expect("lcm state poisoned");
            Snapshot {
                session_id: effective_session(&state.session_id),
                tokenizer: state.tokenizer.clone(),
                threshold_tokens: state.threshold_tokens,
                context_length: state.context_length,
                last_prompt_tokens: state.last_prompt_tokens,
                compaction_count: state.compaction_count,
                session_ignored: state.session_ignored,
                session_stateless: state.session_stateless,
                usage: state.usage,
                model: state.model.clone(),
                context_length_source: state.context_length_source.clone(),
                last_compression_status: if state.last_compression_status.is_empty() {
                    "idle".to_string()
                } else {
                    state.last_compression_status.clone()
                },
                last_compression_noop_reason: state.last_compression_noop_reason.clone(),
                ingest_reconciliation: state.last_ingest_reconciliation.clone(),
            }
        };
        let cx = ToolCx {
            store: &self.store,
            config: &self.config,
            tokenizer: &snap.tokenizer,
            aux: self.aux.as_ref(),
            session_id: &snap.session_id,
            threshold_tokens: snap.threshold_tokens,
            context_length: snap.context_length,
            last_prompt_tokens: snap.last_prompt_tokens,
            compaction_count: snap.compaction_count,
            session_ignored: snap.session_ignored,
            session_stateless: snap.session_stateless,
            ignored_message_count: ignored_message_count(),
            usage: snap.usage,
            model: &snap.model,
            context_length_source: &snap.context_length_source,
            last_compression_status: &snap.last_compression_status,
            last_compression_noop_reason: &snap.last_compression_noop_reason,
            ingest_reconciliation: &snap.ingest_reconciliation,
        };
        crate::tools::dispatch(&cx, name, args).await
    }

    /// Sanitize the provider-facing conversation in place (the active-replay side of §8,
    /// `LCM:engine.py:3243-3289`): text redaction through the active sensitive catalog and
    /// quarantine of runaway assistant output. Ignored/stateless sessions get redaction only
    /// (Python's early return); a turn matching `ignore_message_patterns` is quarantined with the
    /// volatile placeholder (its content must never touch disk); everything else spills the
    /// quarantined body to the externalization dir so it stays recoverable.
    fn sanitize_active_replay(&self, conv: &mut Conversation) {
        let active: &[String] = if self.config.sensitive_patterns_enabled {
            &self.config.sensitive_patterns
        } else {
            &[]
        };
        let (filtered, session) = {
            let state = self.state.lock().expect("lcm state poisoned");
            (
                state.session_ignored || state.session_stateless,
                effective_session(&state.session_id),
            )
        };
        let ext_dir = self.config.externalization_dir();
        for turn in conv.turns.iter_mut() {
            let quarantine = if filtered {
                ReplayQuarantine::Skip
            } else if !self.message_patterns.is_empty()
                && self.message_patterns.is_match(&turn_match_text(turn))
            {
                ReplayQuarantine::Volatile
            } else {
                match ext_dir.as_deref() {
                    Some(dir) => ReplayQuarantine::Spill(dir),
                    None => ReplayQuarantine::Volatile,
                }
            };
            sanitize_replay_turn(turn, active, &session, quarantine);
        }
    }

    /// Test/diagnostic access to the underlying store.
    #[cfg(test)]
    pub(crate) fn store(&self) -> &Store {
        &self.store
    }

    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Mirror the live transcript into the `messages` store so the `lcm_*` tools (grep/expand) see
    /// the whole conversation, not just compacted spans. Idempotent and incremental within an
    /// incarnation (only `turns[cursor..]` are ingested); on the first call of an incarnation it
    /// reconciles the volatile tail (`store_id > frontier`) against the replayed conversation by
    /// deleting it and re-ingesting, so a rehydrated session never duplicates rows.
    fn ingest_current(&self, conv: &Conversation) {
        let now = Self::now();
        let mut state = self.state.lock().expect("lcm state poisoned");
        // Ignored/stateless sessions never write to the store (§12.5).
        if state.session_ignored || state.session_stateless {
            return;
        }
        let session = effective_session(&state.session_id);
        let tok = state.tokenizer.clone();
        if !state.reconciled {
            state.reconciled = true;
            let session_count = self.store.message_count(&session).unwrap_or(0);
            if session_count <= 0 {
                // Fresh session — nothing durable yet, ingest from the top.
                state.cursor = 0;
                state.turn_store_ids.clear();
            } else if leading_scaffold_count(&conv.turns) > 0 {
                // A genuine LCM-compacted replay leads with a summary scaffold turn: the volatile
                // tail (`store_id > frontier`) is re-supplied by the replay, so delete it and
                // rebuild the turn→row index by re-ingesting from the top (the summarized head
                // stays behind the frontier). This is the original reconcile path, retained for
                // the compaction-restart case.
                let frontier = self.store.get_frontier(&session).unwrap_or(0);
                let _ = self.store.delete_messages_after(&session, frontier);
                state.cursor = 0;
                state.turn_store_ids.clear();
            } else {
                // Non-scaffold restart: match the replayed prefix against the durable store tail
                // and advance the cursor past it WITHOUT deleting durable rows (a core port of
                // `_reconcile_ingest_cursor_from_store`, `LCM:engine.py:3125`). A delta, a stale
                // short snapshot, or a scaffold+new-rows replay must not wipe or duplicate the
                // durable transcript.
                // A plain system prompt (no LCM note) is the Rust anchor for Python's
                // system-message stale-snapshot guard.
                let system_is_plain = !conv
                    .system
                    .text
                    .contains("Lossless Context Management (LCM)");
                let (cursor, kind, record) = self.reconcile_turn_cursor(
                    &session,
                    &conv.turns,
                    &tok,
                    session_count,
                    system_is_plain,
                );
                let cursor = cursor.min(conv.turns.len());
                state.turn_store_ids =
                    self.rebuild_turn_store_ids(&session, &conv.turns, cursor, &tok, kind);
                state.cursor = cursor;
                if let Some(record) = record {
                    state.last_ingest_reconciliation = record;
                }
            }
        }
        let scaffold = leading_scaffold_count(&conv.turns);
        let ext_dir = self.config.externalization_dir();
        while state.cursor < conv.turns.len() {
            let idx = state.cursor;
            if idx < scaffold {
                // A synthetic summary scaffold turn carries no real messages.
                state.turn_store_ids.push(Vec::new());
            } else if !self.message_patterns.is_empty()
                && self
                    .message_patterns
                    .is_match(&turn_match_text(&conv.turns[idx]))
            {
                // §12.3 ignore filter: drop this turn (keep index alignment with an empty slot) and
                // bump the process-lifetime ignored counter.
                state.turn_store_ids.push(Vec::new());
                IGNORED_MESSAGE_COUNT.fetch_add(1, Ordering::Relaxed);
            } else {
                let rows = flatten_turns(std::slice::from_ref(&conv.turns[idx]), &tok);
                // §8 ingest protection: redact/quarantine/externalize at the write boundary before
                // the rows hit `messages` (the storage guard no-ops for an ephemeral bank).
                let rows: Vec<_> = rows
                    .into_iter()
                    .map(|m| {
                        protect_message_for_ingest(m, &self.config, &session, ext_dir.as_deref())
                    })
                    .collect();
                let ids = self
                    .store
                    .append_batch(&session, &rows, now)
                    .unwrap_or_default();
                state.turn_store_ids.push(ids);
            }
            state.cursor += 1;
        }
    }

    /// Whether a durable store row matches the active `ignore_message_patterns` (so it is excluded
    /// from tail reconciliation, mirroring Python's `stored_row=True` filtering).
    fn row_is_ignored(&self, row: &crate::store::MessageRow) -> bool {
        !self.message_patterns.is_empty()
            && self
                .message_patterns
                .is_match(row.content.as_deref().unwrap_or(""))
    }

    /// Whether a live turn matches the active `ignore_message_patterns` (§12.3).
    fn turn_is_ignored(&self, turn: &Turn) -> bool {
        !self.message_patterns.is_empty() && self.message_patterns.is_match(&turn_match_text(turn))
    }

    /// Restart reconcile for a non-scaffold replay: return `(turn_cursor, kind)` — how many leading
    /// turns of `turns` are a replay of the durable store, and why. A core port of
    /// `_reconcile_ingest_cursor_from_store` + `_find_reconciled_cursor_for_store_tail`
    /// (`LCM:engine.py:2910-3222`), operating at turn granularity (each turn flattens to one or
    /// more message identities). Never deletes durable rows.
    ///
    /// Stored-row identities restore externalized ingest payloads before matching (the
    /// `stored_row=True` restoration in `_message_replay_identity`), so a replay carrying the raw
    /// inline payload still matches the placeholder row the write boundary persisted. The
    /// stale-snapshot proof compares against the RAW durable prefix (ignore filters may suppress
    /// noisy rows for tail reconciliation, but filtered history must not create replay evidence
    /// for skipping a batch — `LCM:engine.py:3186-3192`).
    ///
    /// Adaptation (see PARITY.md): the Python system-message anchor for the stale-snapshot guard is
    /// mapped to "the conversation carries a plain system prompt (no LCM note)", because the Rust
    /// `Conversation` keeps the system prompt out of the turn/row stream. Quarantine-identity
    /// restoration and the sanitized-active-cleanup equivalence flags are not modeled here
    /// (documented gaps).
    fn reconcile_turn_cursor(
        &self,
        session: &str,
        turns: &[Turn],
        tok: &Tokenizer,
        session_count: i64,
        system_is_plain: bool,
    ) -> (usize, ReconcileKind, Option<Value>) {
        let all_rows = self.store.session_messages(session).unwrap_or_default();
        let ext_dir = self.config.externalization_dir();
        // Durable identities oldest→newest: `stored_raw_all` keeps every row (the stale-proof
        // view); `stored_all` is the ignore-filtered view used for tail matching.
        let stored_raw_all: Vec<ReplayId> = all_rows
            .iter()
            .map(|r| row_replay_id(r, ext_dir.as_deref()))
            .collect();
        let stored_all: Vec<ReplayId> = all_rows
            .iter()
            .zip(stored_raw_all.iter())
            .filter(|(r, _)| !self.row_is_ignored(r))
            .map(|(_, id)| id.clone())
            .collect();
        if stored_all.is_empty() {
            return (0, ReconcileKind::AmbiguousDelta, None);
        }
        let flat_incoming: usize = turns
            .iter()
            .map(|t| flatten_turns(std::slice::from_ref(t), tok).len())
            .sum();
        let tail_limit = (flat_incoming.saturating_mul(4).max(64)).min(stored_all.len());
        let stored_tail = &stored_all[stored_all.len() - tail_limit..];

        // Precompute per-turn flags/identities once.
        let per_turn: Vec<(bool, bool, Vec<ReplayId>)> = turns
            .iter()
            .map(|t| {
                (
                    is_scaffold_turn(t),
                    self.turn_is_ignored(t),
                    turn_replay_ids(t, tok),
                )
            })
            .collect();
        let visible_prefix = |upto: usize| -> Vec<ReplayId> {
            let mut out = Vec::new();
            for (scaffold, ignored, ids) in per_turn.iter().take(upto) {
                if !*scaffold && !*ignored {
                    out.extend(ids.iter().cloned());
                }
            }
            out
        };

        let n = turns.len();
        let effective_session_count = stored_tail.len();
        // The `_record_ingest_reconciliation` analog (`LCM:engine.py:3065-3085`); counts are
        // turn-based where Python counts messages (adaptation).
        let record = |action: &str, reason: &str, cursor: usize, effective_incoming: usize| {
            serde_json::json!({
                "action": action,
                "reason": reason,
                "cursor": cursor,
                "incoming": n,
                "session_count": session_count,
                "stored_tail_count": stored_tail.len(),
                "effective_incoming": effective_incoming,
            })
        };
        // Search high→low for the largest turn cursor proving a durable-tail replay.
        for t in (0..=n).rev() {
            let candidate_prefix = visible_prefix(t);
            if candidate_prefix.is_empty() {
                // A leading run of scaffold/ignored-only turns; skip it (allow-empty-prefix).
                if t > 0 {
                    let rec = record(
                        "advanced cursor",
                        "skipped scaffold-only prefix",
                        t,
                        visible_prefix(n).len(),
                    );
                    return (t, ReconcileKind::ScaffoldOnly, Some(rec));
                }
                break;
            }
            if !matches_store_tail_suffix(stored_tail, &candidate_prefix) {
                continue;
            }
            let has_scaffold_evidence = per_turn.iter().take(t).any(|(s, _, _)| *s);
            let has_effective_full_replay =
                candidate_prefix.len() >= effective_session_count && effective_session_count > 1;
            let has_raw_full_replay = !has_scaffold_evidence
                && candidate_prefix.len() >= session_count as usize
                && session_count > 1;
            if has_effective_full_replay || has_raw_full_replay {
                let rec = record(
                    "advanced cursor",
                    "replayed durable tail",
                    t,
                    visible_prefix(n).len(),
                );
                return (t, ReconcileKind::TailReplay, Some(rec));
            }
        }

        // No qualifying replay cursor. A stale short snapshot that exactly re-supplies the durable
        // head with no tail overlap must be skipped, not appended (a duplicate guard). The proof
        // uses the RAW durable prefix — filtered rows must not manufacture stale evidence.
        // Otherwise the batch is an ambiguous delta and is persisted from the top.
        let incoming = visible_prefix(n);
        if is_suspicious_stale_no_overlap(&incoming, stored_tail, &stored_raw_all, system_is_plain)
        {
            let rec = record(
                "skipped batch",
                "skipped stale no-overlap snapshot",
                n,
                incoming.len(),
            );
            return (n, ReconcileKind::StaleSkip, Some(rec));
        }
        let rec = record(
            "persisted batch",
            "persisted ambiguous delta",
            0,
            incoming.len(),
        );
        (0, ReconcileKind::AmbiguousDelta, Some(rec))
    }

    /// Rebuild `turn_store_ids` for the reconciled replay prefix `turns[0..cursor]` from the durable
    /// store rows (so a later compaction still maps replayed turns to their real `store_id`s). The
    /// visible replayed turns correspond to the last `M` durable rows; scaffold/ignored/skip turns
    /// get an empty slot. The result length is exactly `cursor`.
    fn rebuild_turn_store_ids(
        &self,
        session: &str,
        turns: &[Turn],
        cursor: usize,
        tok: &Tokenizer,
        kind: ReconcileKind,
    ) -> Vec<Vec<i64>> {
        if cursor == 0 {
            return Vec::new();
        }
        // For a stale skip the incoming prefix matches the durable head, not the tail — there is no
        // sound tail mapping, so leave the slots empty (a skipped snapshot is never compacted).
        if matches!(kind, ReconcileKind::StaleSkip) {
            return vec![Vec::new(); cursor];
        }
        let all_rows = self.store.session_messages(session).unwrap_or_default();
        let store_ids: Vec<i64> = all_rows.iter().map(|r| r.store_id).collect();
        let flat_count = |t: &Turn| flatten_turns(std::slice::from_ref(t), tok).len();
        let covered: usize = turns
            .iter()
            .take(cursor)
            .filter(|t| !is_scaffold_turn(t) && !self.turn_is_ignored(t))
            .map(flat_count)
            .sum();
        let start = store_ids.len().saturating_sub(covered);
        let tail_ids = &store_ids[start..];
        let mut ptr = 0usize;
        let mut out = Vec::with_capacity(cursor);
        for t in turns.iter().take(cursor) {
            if is_scaffold_turn(t) || self.turn_is_ignored(t) {
                out.push(Vec::new());
                continue;
            }
            let f = flat_count(t);
            let slice: Vec<i64> = tail_ids.iter().skip(ptr).take(f).copied().collect();
            ptr += f;
            out.push(slice);
        }
        out
    }
}

/// Why a non-scaffold restart reconcile advanced (or held) the ingest cursor
/// (`_record_ingest_reconciliation` reasons, `LCM:engine.py:3065`).
#[derive(Clone, Copy)]
enum ReconcileKind {
    /// The prefix replays the durable tail — advance past it.
    TailReplay,
    /// A leading scaffold-only prefix — skip it.
    ScaffoldOnly,
    /// A stale short snapshot re-supplying the durable head with no tail overlap — skip the batch.
    StaleSkip,
    /// No replay proof — persist the batch from the top.
    AmbiguousDelta,
}

/// A message-level replay identity for restart reconciliation (a core port of
/// `_message_replay_identity`, `LCM:engine.py:2754`): `(role, content, tool_call_id, tool_calls)`.
/// Stored rows restore externalized ingest payloads (`stored_row=True`); quarantine restoration is
/// not modeled (adaptation — see PARITY.md).
#[derive(Clone, PartialEq, Eq)]
struct ReplayId {
    role: String,
    content: String,
    tool_call_id: String,
    tool_calls: String,
}

/// The replay identity of a durable store row. Externalized ingest-payload placeholders are
/// restored to their original inline content (the `stored_row=True` branch of
/// `_message_replay_identity`) so a restart replay carrying the raw payload matches the row the
/// write boundary persisted; quarantined-assistant spills keep their placeholder (kind mismatch).
fn row_replay_id(row: &crate::store::MessageRow, ext_dir: Option<&std::path::Path>) -> ReplayId {
    let mut content = row.content.clone().unwrap_or_default();
    if let Some(dir) = ext_dir {
        if content.contains("[Externalized LCM ingest payload:") {
            content =
                crate::externalize::restore_ingest_placeholders(dir, &content, &row.session_id);
        }
    }
    ReplayId {
        role: row.role.clone(),
        content,
        tool_call_id: row.tool_call_id.clone().unwrap_or_default(),
        tool_calls: row.tool_calls.clone().unwrap_or_default(),
    }
}

/// Flatten one turn into its message-level replay identities (mirrors [`flatten_turns`]).
fn turn_replay_ids(turn: &Turn, tok: &Tokenizer) -> Vec<ReplayId> {
    flatten_turns(std::slice::from_ref(turn), tok)
        .into_iter()
        .map(|m| ReplayId {
            role: m.role,
            content: m.content.unwrap_or_default(),
            tool_call_id: m.tool_call_id.unwrap_or_default(),
            tool_calls: m.tool_calls.unwrap_or_default(),
        })
        .collect()
}

/// Whether a turn is replayed active-context scaffolding that must not be re-ingested (a port of
/// `_is_replayed_context_scaffold_message`, `LCM:engine.py:2644`, at turn granularity — the
/// system-prompt branch is handled out-of-band since Rust keeps the system prompt off the turn
/// stream).
fn is_scaffold_turn(turn: &Turn) -> bool {
    let text = match turn {
        Turn::User(u) => u.text.as_str(),
        Turn::Assistant(a) => a.text.as_str(),
        Turn::Tool(t) => t.assistant.text.as_str(),
    };
    let trimmed = text.trim_start();
    if trimmed.starts_with(crate::compaction::SUMMARY_SENTINEL)
        || trimmed.starts_with(crate::compaction::PRESERVED_OBJECTIVE_PREFIX)
    {
        return true;
    }
    if !text.contains("[Expand for details:") {
        return false;
    }
    summary_header_regex().is_match(text)
}

/// The synthetic-summary header pattern (`LCM:engine.py:2658`).
fn summary_header_regex() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
        )
        .expect("summary header regex is valid")
    })
}

/// `stored_tail[-len(prefix):] == prefix` (a port of `_matches_store_tail_suffix`,
/// `LCM:engine.py:2782`).
fn matches_store_tail_suffix(stored_tail: &[ReplayId], candidate_prefix: &[ReplayId]) -> bool {
    if candidate_prefix.is_empty() {
        return true;
    }
    if candidate_prefix.len() > stored_tail.len() {
        return false;
    }
    &stored_tail[stored_tail.len() - candidate_prefix.len()..] == candidate_prefix
}

/// A short stale snapshot that exactly re-supplies the durable head with no tail overlap (a port of
/// `_is_suspicious_stale_no_overlap_snapshot`, `LCM:engine.py:3098`). The Python system-role anchor
/// is adapted to `system_is_plain` — the incoming carries a plain (non-LCM) system prompt.
fn is_suspicious_stale_no_overlap(
    incoming: &[ReplayId],
    stored_tail: &[ReplayId],
    stored_head: &[ReplayId],
    system_is_plain: bool,
) -> bool {
    if incoming.len() <= 1 {
        return false;
    }
    if !system_is_plain {
        return false;
    }
    if stored_tail.is_empty() || incoming.len() >= stored_tail.len() {
        return false;
    }
    if incoming.iter().any(|m| stored_tail.contains(m)) {
        return false;
    }
    if incoming.len() > stored_head.len() {
        return false;
    }
    &stored_head[..incoming.len()] == incoming
}

/// The matchable text of a turn for the `ignore_message_patterns` filter (§12.3): user/assistant
/// text, or a tool turn's assistant text plus its result bodies. Mirrors
/// [`protection::text_content_for_pattern_matching`](crate::protection) at the turn level.
fn turn_match_text(turn: &Turn) -> String {
    match turn {
        Turn::User(u) => u.text.clone(),
        Turn::Assistant(a) => a.text.clone(),
        Turn::Tool(t) => {
            let mut s = t.assistant.text.clone();
            for (_, result) in &t.calls {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&result.content);
            }
            s
        }
    }
}

/// Normalize an unset session id to a stable placeholder so store rows are attributable.
fn effective_session(session_id: &str) -> String {
    if session_id.is_empty() {
        "unknown".to_string()
    } else {
        session_id.to_string()
    }
}

#[async_trait]
impl ContextEngine for LcmContextEngine {
    fn on_model(&self, model: &ModelInfo) {
        let mut state = self.state.lock().expect("lcm state poisoned");
        state.tokenizer = Tokenizer::for_model(&model.model);
        state.model = model.model.clone();
        if let Some(max) = model.max_context {
            state.context_length = Some(max as usize);
            state.threshold_tokens = Some((max as f64 * self.config.context_threshold) as usize);
            state.context_length_source = "model_info".to_string();
        }
    }

    /// The LCM tooling note, contributed as a static guidance slot from session start — the
    /// cache-correct home of the former first-compaction `Conversation.system` append
    /// (`_append_lcm_note_to_content`): the composed prefix never mutates mid-session.
    fn guidance_block(&self) -> Option<String> {
        Some(crate::compaction::LCM_SYSTEM_NOTE.to_string())
    }

    fn before_turn(&self, conv: &mut Conversation, budget: Option<usize>) -> Pressure {
        // Keep the durable transcript current before measuring pressure (so the tools see this
        // turn) — the store ingests the *original* content (protected at the write boundary).
        self.ingest_current(conv);
        // Active-replay protection (`LCM:engine.py:3224-3289`): sanitize the provider-facing
        // conversation in place — sensitive redaction over every turn plus quarantine of runaway
        // assistant output — then measure pressure on what will actually be sent.
        self.sanitize_active_replay(conv);
        let mut state = self.state.lock().expect("lcm state poisoned");
        let used_tokens = state.tokenizer.count_conversation(conv);
        // Remember the measured prompt size so `lcm_doctor`'s `context_pressure` check can report
        // live usage vs the compaction threshold.
        state.last_prompt_tokens = used_tokens;
        // Ignored/stateless sessions are never compacted (no store writes) — report no budget so the
        // turn loop never calls `compact` for them (§12.5).
        let filtered = state.session_ignored || state.session_stateless;
        // Boundary cooldown: after a no-op compaction, report no budget for a short window so the
        // engine doesn't re-attempt a compaction it can't make progress on every turn.
        let in_cooldown = matches!(state.last_noop_at, Some(t) if t.elapsed() < BOUNDARY_COOLDOWN);
        if in_cooldown || filtered {
            return Pressure {
                used_tokens,
                budget_tokens: None,
            };
        }
        let mut budget_tokens = budget.or(state.threshold_tokens);
        let session = effective_session(&state.session_id);
        let critical =
            critical_budget_pressure_reached(&self.config, state.context_length, used_tokens);

        // Cap-driven force overflow (`_should_force_overflow_recovery`, `LCM:engine.py:657/681`):
        // at or over the configured assembly cap, always advertise compaction targeting the cap —
        // overflow recovery bypasses every polite gate below.
        let assembly_cap = self
            .config
            .effective_assembly_token_cap(state.context_length);
        if let Some(cap) = assembly_cap.filter(|cap| used_tokens >= *cap) {
            let target = cap.saturating_sub(1).max(1);
            return Pressure {
                used_tokens,
                budget_tokens: Some(budget_tokens.unwrap_or(usize::MAX).min(target)),
            };
        }

        let over_threshold = matches!(budget_tokens, Some(b) if used_tokens > b);
        if over_threshold {
            // Leaf-floor gate (`_leaf_compaction_candidate_status`, `LCM:engine.py:684-733`): a
            // session can be over threshold while all pressure sits in the protected fresh tail
            // or the backlog is below one working leaf chunk — `compact()` would immediately
            // no-op, so don't advertise the pressure (unless a debt-carrying turn wants a
            // deferred catch-up pass). Error-driven overflow recovery calls `compact` directly
            // and bypasses this.
            let (eligible, reason) =
                leaf_compaction_candidate_status(&conv.turns, &state.tokenizer, &self.config);
            if !eligible
                && !should_run_deferred_maintenance(
                    &self.store,
                    &state.tokenizer,
                    &self.config,
                    &session,
                    &conv.turns,
                    critical,
                )
            {
                tracing::info!(
                    used = used_tokens,
                    reason,
                    "lcm: preflight compression no-op"
                );
                state.last_compression_status = "noop".to_string();
                state.last_compression_noop_reason = reason.to_string();
                budget_tokens = None;
            }
        } else {
            // Under threshold (or no threshold yet): keep the lifecycle debt current and let a
            // debt-carrying conversation trigger a deferred catch-up pass anyway
            // (`should_compress_preflight`, `LCM:engine.py:693-694`).
            refresh_raw_backlog_debt(
                &self.store,
                &state.tokenizer,
                &self.config,
                &session,
                &conv.turns,
                critical,
                Self::now(),
            );
            if should_run_deferred_maintenance(
                &self.store,
                &state.tokenizer,
                &self.config,
                &session,
                &conv.turns,
                critical,
            ) {
                budget_tokens = Some(used_tokens.saturating_sub(1).max(1));
            }
        }
        Pressure {
            used_tokens,
            budget_tokens,
        }
    }

    async fn compact(&self, conv: Conversation, budget: usize) -> Conversation {
        // Catch up the ingest index to the live conversation (the ReAct loop may have appended turns
        // since `before_turn`), then snapshot the bits compaction needs and run it without holding
        // the state lock across the aux-provider `await`s. The breaker + index are taken out and
        // restored afterwards.
        // Ignored/stateless sessions are read-only: never write summary nodes / advance the
        // frontier (`LCM:engine.py:870-885` — status reports the bypass).
        {
            let mut state = self.state.lock().expect("lcm state poisoned");
            if state.session_ignored || state.session_stateless {
                let reason = if state.session_ignored {
                    "bypassed: ignored session"
                } else {
                    "bypassed: stateless session"
                };
                state.last_compression_status = "noop".to_string();
                state.last_compression_noop_reason = reason.to_string();
                return conv;
            }
        }
        self.ingest_current(&conv);
        let (tokenizer, session_id, mut breakers, index, context_length) = {
            let mut state = self.state.lock().expect("lcm state poisoned");
            let breakers = std::mem::take(&mut state.breakers);
            let index = std::mem::take(&mut state.turn_store_ids);
            (
                state.tokenizer.clone(),
                state.session_id.clone(),
                breakers,
                index,
                state.context_length,
            )
        };
        let session = effective_session(&session_id);

        let outcome = run_compaction(
            &self.store,
            &tokenizer,
            &self.config,
            &self.aux_chain,
            &mut breakers,
            &session,
            index,
            conv,
            budget,
            context_length,
            Self::now(),
        )
        .await;

        let mut state = self.state.lock().expect("lcm state poisoned");
        state.breakers = breakers;
        state.cursor = outcome.index.len();
        state.turn_store_ids = outcome.index;
        match outcome.status {
            crate::compaction::CompressionStatus::Compacted => {
                state.compaction_count += 1;
                state.last_noop_at = None;
                state.last_compression_status = "compacted".to_string();
                state.last_compression_noop_reason.clear();
            }
            // The context changed (no new summary node) — no cooldown: pressure re-measures on
            // the recovered context next turn.
            crate::compaction::CompressionStatus::OverflowRecovery => {
                state.last_noop_at = None;
                state.last_compression_status = "overflow_recovery".to_string();
                state.last_compression_noop_reason.clear();
            }
            crate::compaction::CompressionStatus::Noop(reason) => {
                state.last_noop_at = Some(Instant::now());
                state.last_compression_status = "noop".to_string();
                state.last_compression_noop_reason = reason;
                tracing::info!(
                    reason = %state.last_compression_noop_reason,
                    "lcm: compression no-op"
                );
            }
        }
        outcome.conv
    }

    fn on_session_start(&self, session: &SessionId) {
        // Switching sessions finalizes the previous binding only when a reset boundary is pending
        // for it (`_finalize_pending_reset_boundary`, `LCM:engine.py:2115-2123`); either way the
        // runtime state resets so the next ingest reconciles against the durable frontier.
        let (previous, pending_reset) = {
            let mut state = self.state.lock().expect("lcm state poisoned");
            let previous = std::mem::replace(&mut state.session_id, session.as_str().to_string());
            let pending_reset = state.pending_reset.take();
            Self::reset_session_scoped_runtime_state(&mut state);
            self.refresh_session_filters(session.as_str(), &mut state);
            (previous, pending_reset)
        };
        if !previous.is_empty() && previous != session.as_str() {
            if let Some((reset_session, reset_frontier)) = pending_reset {
                if reset_session == previous {
                    let frontier =
                        reset_frontier.max(self.store.get_frontier(&previous).unwrap_or(0));
                    let _ =
                        self.store
                            .finalize_session(&previous, &previous, frontier, Self::now());
                }
            }
        }
        self.bind_lifecycle(session.as_str());
    }

    fn on_session_end(&self, session: &SessionId, conv: &Conversation) {
        // Best-effort final flush so the last turns are durable (`LCM:engine.py:2132-2200`), then
        // mark the session finalized in the lifecycle row (frontier = the durable high-water mark).
        self.ingest_current(conv);
        let frontier = self.store.get_frontier(session.as_str()).unwrap_or(0);
        let _ =
            self.store
                .finalize_session(session.as_str(), session.as_str(), frontier, Self::now());
        let count = self.store.summary_count(session.as_str()).unwrap_or(0);
        tracing::debug!(session = %session, summaries = count, "lcm: session ended");
    }

    /// The `/new`-style reset hook (fired by the engine on a full-clear rewind): delegate to
    /// [`Self::reset_bound_session`]. Defensive: this engine instance is per-session, so a reset
    /// for a *different* session indicates a wiring bug — warn and skip rather than pruning the
    /// wrong session's DAG. An unbound engine (no turn ran yet) has nothing to reset.
    fn on_session_reset(&self, session: &SessionId) {
        let bound = {
            let state = self.state.lock().expect("lcm state poisoned");
            state.session_id.clone()
        };
        if !bound.is_empty() && bound != session.as_str() {
            tracing::warn!(
                bound = %bound,
                requested = %session,
                "lcm: session reset for a session this engine is not bound to; skipping"
            );
            return;
        }
        self.reset_bound_session();
    }

    /// The old → new rollover hook: delegate to [`Self::rollover_sessions`] (which already guards
    /// against an unbound/mismatched old session), dropping the carried-node count.
    fn rollover_session(
        &self,
        old: &SessionId,
        new: &SessionId,
        old_conv: Option<&Conversation>,
        carry_over: bool,
    ) {
        self.rollover_sessions(old.as_str(), new.as_str(), old_conv, carry_over);
    }

    /// Record provider-reported usage from the last model response (`update_from_response`,
    /// `LCM:engine.py:614-629`) — surfaced by `lcm_status`/`lcm_doctor`. Provider-reported prompt
    /// tokens override the `before_turn` estimate when present (`0` means the provider did not
    /// report, so the measured estimate stands).
    fn after_response(&self, usage: &daemon_common::UsageDelta) {
        let mut state = self.state.lock().expect("lcm state poisoned");
        state.usage = UsageMetrics {
            last_input_tokens: usage.input_tokens,
            last_output_tokens: usage.output_tokens,
            last_cache_read_tokens: usage.cache_read_tokens,
            last_cache_write_tokens: usage.cache_write_tokens,
            last_reasoning_tokens: usage.reasoning_tokens,
            cache_metrics_available: usage.cache_read_tokens > 0 || usage.cache_write_tokens > 0,
        };
        if usage.input_tokens > 0 {
            state.last_prompt_tokens = usage.input_tokens as usize;
        }
    }

    /// The advisory names of the `lcm_*` tools this engine owns; the host registers their actual
    /// dispatch through the §12 [`ToolRegistry`](daemon_core::tools) (see [`Self::tool_defs`]).
    fn tools(&self) -> Vec<String> {
        TOOL_NAMES.iter().map(|s| s.to_string()).collect()
    }

    /// Expose this engine as a [`CommandProvider`] so the node command registry folds in `/lcm`
    /// (the operator maintenance surface, the port of `hermes-lcm`'s `command.py`).
    fn command_provider(self: Arc<Self>) -> Option<CommandProviderHandle> {
        Some(self)
    }
}

/// The result of one `/lcm backup`-style snapshot write.
struct BackupResult {
    /// The snapshot destination.
    path: PathBuf,
    /// Its on-disk size in bytes.
    size: u64,
}

/// The rotate preview/apply outcome (`rotate_active_session`, `LCM:engine.py:4453-4620`), shared
/// by the preview and apply formatters.
struct RotateOutcome {
    /// `false` = refused (see `reason`); nothing was or would be changed.
    ok: bool,
    /// `frontier_already_ahead` / `no_pre_tail_content` / `empty_tail` / a refusal code.
    reason: String,
    /// Whether an apply would change (or changed) nothing.
    noop: bool,
    session_id: String,
    total_message_count: i64,
    fresh_tail_count: usize,
    pre_tail_message_count: i64,
    current_frontier_store_id: i64,
    new_frontier_store_id: i64,
}

impl LcmContextEngine {
    /// Write a timestamped snapshot of the live database under the bank's backup directory
    /// (`_backup_database`, `LCM:command.py:454-489`). Timestamps are UTC (the Python plugin
    /// stamps local time).
    fn backup_database(&self) -> std::result::Result<BackupResult, String> {
        let db_path = self
            .config
            .db_path()
            .filter(|p| p.exists())
            .ok_or_else(|| "database file does not exist".to_string())?;
        let dir = self
            .config
            .backup_dir()
            .ok_or_else(|| "database file does not exist".to_string())?;
        let (y, mo, d, hh, mi, ss) = crate::extraction::civil_datetime(Self::now());
        let stem = db_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "lcm".to_string());
        let dest = dir.join(format!(
            "{stem}-{y:04}{mo:02}{d:02}_{hh:02}{mi:02}{ss:02}.sqlite3"
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        self.store.backup_to(&dest).map_err(|e| e.to_string())?;
        let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        Ok(BackupResult { path: dest, size })
    }

    /// Overwrite the rolling rotate-latest snapshot slot, atomically via tmp-then-rename
    /// (`_rotate_backup_database`, `LCM:command.py:492-545`).
    fn rotate_backup_database(&self) -> std::result::Result<BackupResult, String> {
        if !self.config.db_path().is_some_and(|p| p.exists()) {
            return Err("database file does not exist".to_string());
        }
        let dest = self
            .config
            .rotate_backup_path()
            .ok_or_else(|| "database file does not exist".to_string())?;
        let dir = dest.parent().expect("rotate slot has a parent");
        let tmp = dest.with_file_name(format!(
            "{}.tmp",
            dest.file_name().unwrap_or_default().to_string_lossy()
        ));
        let write = || -> std::result::Result<(), String> {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            if tmp.exists() {
                std::fs::remove_file(&tmp).map_err(|e| e.to_string())?;
            }
            self.store.backup_to(&tmp).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())
        };
        if let Err(e) = write() {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        Ok(BackupResult { path: dest, size })
    }

    /// Tail-preserving in-place compact of the active session (`rotate_active_session`,
    /// `LCM:engine.py:4453-4620`): advance the persisted lifecycle frontier past the pre-tail raw
    /// messages so they stop replaying into a fresh bootstrap. Read-only preview unless `apply`.
    /// Raw rows stay in the store (recoverable via `lcm_load_session`/`lcm_expand`), and the
    /// monotonic `advance_frontier` upsert makes Python's `stale_lifecycle_state` refusal
    /// unreachable here.
    fn rotate_active_session(&self, apply: bool) -> RotateOutcome {
        let (session_id, ignored, stateless) = {
            let state = self.state.lock().expect("lcm state poisoned");
            (
                state.session_id.clone(),
                state.session_ignored,
                state.session_stateless,
            )
        };
        let refuse = |session_id: &str, reason: &str| RotateOutcome {
            ok: false,
            reason: reason.to_string(),
            noop: false,
            session_id: session_id.to_string(),
            total_message_count: 0,
            fresh_tail_count: 0,
            pre_tail_message_count: 0,
            current_frontier_store_id: 0,
            new_frontier_store_id: 0,
        };
        if session_id.is_empty() {
            return refuse("", "no_active_session");
        }
        if ignored {
            return refuse(&session_id, "session_ignored");
        }
        if stateless {
            return refuse(&session_id, "session_stateless");
        }
        let session = effective_session(&session_id);
        let fresh_tail_count = self.config.fresh_tail_count.max(1);
        let total = self.store.message_count(&session).unwrap_or(0);
        let current_frontier = self.store.get_frontier(&session).unwrap_or(0);
        let mut out = RotateOutcome {
            ok: true,
            reason: String::new(),
            noop: true,
            session_id: session,
            total_message_count: total,
            fresh_tail_count,
            pre_tail_message_count: 0,
            current_frontier_store_id: current_frontier,
            new_frontier_store_id: current_frontier,
        };
        if total <= fresh_tail_count as i64 {
            out.reason = "no_pre_tail_content".to_string();
            return out;
        }
        let Ok(Some(smallest_tail)) = self
            .store
            .tail_min_store_id(&out.session_id, fresh_tail_count as i64)
        else {
            // Concurrent deletion can empty the tail after the count check.
            out.reason = "empty_tail".to_string();
            return out;
        };
        let new_frontier = (smallest_tail - 1).max(0);
        out.pre_tail_message_count = total - fresh_tail_count as i64;
        out.new_frontier_store_id = new_frontier;
        out.noop = new_frontier <= current_frontier;
        if out.noop {
            out.reason = "frontier_already_ahead".to_string();
            return out;
        }
        if apply {
            let _ = self
                .store
                .advance_frontier(&out.session_id, new_frontier, Self::now());
        }
        out
    }
}

/// Python's `_fmt_size` (`LCM:command.py:40-50`): human-readable bytes with Python's precision
/// ladder (0 decimals >= 100, 1 >= 10, else 2).
fn fmt_size(num_bytes: u64) -> String {
    if num_bytes < 1024 {
        return format!("{num_bytes} B");
    }
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = num_bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    let precision = if value >= 100.0 {
        0
    } else if value >= 10.0 {
        1
    } else {
        2
    };
    format!("{value:.precision$} {}", UNITS[unit])
}

/// `/lcm backup` (`_backup_text`, `LCM:command.py:1542-1559`).
fn backup_text(engine: &LcmContextEngine) -> String {
    let db = engine
        .config
        .db_path()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    match engine.backup_database() {
        Ok(b) => [
            "LCM backup".to_string(),
            "status: ok".to_string(),
            format!("database_path: {db}"),
            format!("backup_path: {}", b.path.display()),
            format!("backup_size: {}", fmt_size(b.size)),
            "note: backup created before any future cleanup/apply workflow".to_string(),
        ]
        .join("\n"),
        Err(e) => [
            "LCM backup".to_string(),
            "status: error".to_string(),
            format!("database_path: {db}"),
            format!("error: {e}"),
        ]
        .join("\n"),
    }
}

/// The shared rotate field block (both the preview and apply renderings).
fn rotate_fields(r: &RotateOutcome) -> Vec<String> {
    vec![
        format!("session_id: {}", r.session_id),
        format!("conversation_id: {}", r.session_id),
        format!("total_message_count: {}", r.total_message_count),
        format!("fresh_tail_count: {}", r.fresh_tail_count),
        format!("pre_tail_message_count: {}", r.pre_tail_message_count),
    ]
}

/// `/lcm rotate` — the read-only preview (`_rotate_text`, `LCM:command.py:546-582`).
fn rotate_text(engine: &LcmContextEngine) -> String {
    let preview = engine.rotate_active_session(false);
    if !preview.ok {
        let mut lines = vec![
            "LCM rotate".to_string(),
            "status: refused".to_string(),
            format!("reason: {}", preview.reason),
        ];
        if !preview.session_id.is_empty() {
            lines.push(format!("session_id: {}", preview.session_id));
        }
        lines.push("note: read-only preview — no changes were made".to_string());
        return lines.join("\n");
    }
    let backup_path = engine
        .config
        .rotate_backup_path()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let mut lines = vec![
        "LCM rotate".to_string(),
        format!("status: {}", if preview.noop { "noop" } else { "preview" }),
    ];
    lines.extend(rotate_fields(&preview));
    lines.push(format!(
        "current_frontier_store_id: {}",
        preview.current_frontier_store_id
    ));
    lines.push(format!(
        "new_frontier_store_id: {}",
        preview.new_frontier_store_id
    ));
    lines.push(format!("rotate_backup_path: {backup_path}"));
    if preview.noop {
        lines.push(format!("reason: {}", preview.reason));
        lines.push(
            "note: read-only preview — rotate apply would be a no-op for this session".to_string(),
        );
    } else {
        lines.push(
            "note: read-only preview — use `/lcm rotate apply` to advance the frontier (backup-first)"
                .to_string(),
        );
        lines.push(
            "note: pre-tail raw messages remain in the store and recoverable via lcm_load_session"
                .to_string(),
        );
    }
    lines.join("\n")
}

/// `/lcm rotate apply` — backup-first frontier advance (`_rotate_apply_text`,
/// `LCM:command.py:585-666`): preview-refusal and noop pre-flights run *before* the rolling
/// backup so a no-op rerun never overwrites the previous known-good snapshot.
fn rotate_apply_text(engine: &LcmContextEngine) -> String {
    let pre = engine.rotate_active_session(false);
    if !pre.ok {
        let mut lines = vec![
            "LCM rotate apply".to_string(),
            "status: refused".to_string(),
            format!("reason: {}", pre.reason),
        ];
        if !pre.session_id.is_empty() {
            lines.push(format!("session_id: {}", pre.session_id));
        }
        lines.push(
            "note: rotate apply refused; no backup was created and no lifecycle state was changed"
                .to_string(),
        );
        return lines.join("\n");
    }
    if pre.noop {
        let mut lines = vec!["LCM rotate apply".to_string(), "status: noop".to_string()];
        lines.extend(rotate_fields(&pre));
        lines.push(format!(
            "previous_frontier_store_id: {}",
            pre.current_frontier_store_id
        ));
        lines.push(format!(
            "new_frontier_store_id: {}",
            pre.new_frontier_store_id
        ));
        lines.push(format!("reason: {}", pre.reason));
        lines.push(
            "note: rotate is a no-op; rolling backup was not written so the previous rotate-latest snapshot is preserved"
                .to_string(),
        );
        return lines.join("\n");
    }
    let backup = match engine.rotate_backup_database() {
        Ok(b) => b,
        Err(e) => {
            let db = engine
                .config
                .db_path()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            return [
                "LCM rotate apply".to_string(),
                "status: error".to_string(),
                format!("database_path: {db}"),
                format!("error: backup failed: {e}"),
                "note: rotate apply aborted before any lifecycle mutation".to_string(),
            ]
            .join("\n");
        }
    };
    let result = engine.rotate_active_session(true);
    let mut lines = vec![
        "LCM rotate apply".to_string(),
        format!("status: {}", if result.noop { "noop" } else { "ok" }),
        format!("session_id: {}", result.session_id),
        format!("conversation_id: {}", result.session_id),
        format!("rotate_backup_path: {}", backup.path.display()),
        format!("rotate_backup_size: {}", fmt_size(backup.size)),
        format!("total_message_count: {}", result.total_message_count),
        format!("fresh_tail_count: {}", result.fresh_tail_count),
        format!("pre_tail_message_count: {}", result.pre_tail_message_count),
        format!(
            "previous_frontier_store_id: {}",
            result.current_frontier_store_id
        ),
        format!("new_frontier_store_id: {}", result.new_frontier_store_id),
    ];
    if result.noop {
        lines.push(format!("reason: {}", result.reason));
        lines.push("note: lifecycle state already at or ahead of the target frontier".to_string());
    } else {
        lines.push(
            "note: pre-tail raw messages remain in the store and recoverable via lcm_load_session"
                .to_string(),
        );
        lines.push("note: rolling backup overwrites the previous rotate-latest slot".to_string());
    }
    lines.join("\n")
}

/// `/lcm doctor source` — read-only scan for legacy blank-`source` rows (`_doctor_source_text`,
/// `LCM:command.py:766`). Reports the source-bucket breakdown and how many rows a normalize would
/// touch, without changing anything.
fn doctor_source_text(engine: &LcmContextEngine) -> String {
    match engine.store.source_normalization_plan() {
        Ok((would_update, affected_sessions, stats)) => {
            let mut lines = vec![
                "LCM doctor source".to_string(),
                format!(
                    "status: {}",
                    if would_update > 0 {
                        "normalization-needed"
                    } else {
                        "ok"
                    }
                ),
                format!("messages_total: {}", stats.messages_total),
                format!("attributed_messages: {}", stats.attributed_messages),
                format!("unknown_messages: {}", stats.normalized_unknown_messages),
                format!(
                    "legacy_blank_messages: {}",
                    stats.legacy_blank_source_messages
                ),
                format!(
                    "effective_unknown_messages: {}",
                    stats.effective_unknown_messages
                ),
                "target_source: unknown".to_string(),
                format!("would_update_messages: {would_update}"),
                format!("affected_sessions: {affected_sessions}"),
                "note: read-only scan only — no source rows were updated".to_string(),
            ];
            if would_update > 0 {
                lines.push(
                    "note: use `/lcm doctor source apply` to create a backup and normalize legacy \
                     blank-source rows"
                        .to_string(),
                );
            } else {
                lines.push("note: no legacy blank-source rows need normalization".to_string());
            }
            lines.join("\n")
        }
        Err(e) => [
            "LCM doctor source".to_string(),
            "status: error".to_string(),
            format!("error: source-lineage scan failed: {e}"),
            "note: read-only scan only — no source rows were updated".to_string(),
        ]
        .join("\n"),
    }
}

/// `/lcm doctor source apply` — backup-first normalization of legacy blank-`source` rows to
/// `unknown` (`_doctor_source_apply_text`, `LCM:command.py:801`). A no-op batch skips the backup;
/// otherwise a timestamped backup is written before the scoped `UPDATE`.
fn doctor_source_apply_text(engine: &LcmContextEngine) -> String {
    let (would_update, _affected, stats_before) = match engine.store.source_normalization_plan() {
        Ok(plan) => plan,
        Err(e) => {
            return [
                "LCM doctor source apply".to_string(),
                "status: error".to_string(),
                format!("error: source-lineage scan failed: {e}"),
                "note: source normalization apply aborted before any rows were updated".to_string(),
            ]
            .join("\n");
        }
    };
    if would_update == 0 {
        return [
            "LCM doctor source apply".to_string(),
            "status: ok".to_string(),
            "target_source: unknown".to_string(),
            "updated_messages: 0".to_string(),
            format!(
                "legacy_blank_before: {}",
                stats_before.legacy_blank_source_messages
            ),
            format!(
                "legacy_blank_after: {}",
                stats_before.legacy_blank_source_messages
            ),
            "note: no legacy blank-source rows needed normalization".to_string(),
        ]
        .join("\n");
    }
    let db = engine
        .config
        .db_path()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let backup = match engine.backup_database() {
        Ok(b) => b,
        Err(e) => {
            return [
                "LCM doctor source apply".to_string(),
                "status: error".to_string(),
                format!("database_path: {db}"),
                format!("error: backup failed: {e}"),
                "note: source normalization apply aborted before any rows were updated".to_string(),
            ]
            .join("\n");
        }
    };
    match engine.store.normalize_legacy_blank_sources() {
        Ok((updated, before, after)) => [
            "LCM doctor source apply".to_string(),
            "status: ok".to_string(),
            format!("database_path: {db}"),
            format!("backup_path: {}", backup.path.display()),
            format!("backup_size: {}", fmt_size(backup.size)),
            "target_source: unknown".to_string(),
            format!("updated_messages: {updated}"),
            format!(
                "legacy_blank_before: {}",
                before.legacy_blank_source_messages
            ),
            format!("legacy_blank_after: {}", after.legacy_blank_source_messages),
            format!("unknown_before: {}", before.normalized_unknown_messages),
            format!("unknown_after: {}", after.normalized_unknown_messages),
            "note: backup created before source normalization apply".to_string(),
        ]
        .join("\n"),
        Err(e) => [
            "LCM doctor source apply".to_string(),
            "status: error".to_string(),
            format!("database_path: {db}"),
            format!("backup_path: {}", backup.path.display()),
            format!("backup_size: {}", fmt_size(backup.size)),
            format!("error: source normalization failed: {e}"),
            "note: backup was created before source normalization apply".to_string(),
        ]
        .join("\n"),
    }
}

/// `/lcm doctor repair` — the read-only FTS repair scan (`_doctor_repair_text`,
/// `LCM:command.py:704-721`).
fn doctor_repair_text(engine: &LcmContextEngine) -> String {
    let scans = engine.store.scan_fts_repair();
    let needs_repair = scans.iter().any(|s| s.needs_repair);
    let mut lines = vec![
        "LCM doctor repair".to_string(),
        format!(
            "status: {}",
            if needs_repair { "repair-needed" } else { "ok" }
        ),
    ];
    for scan in &scans {
        lines.push(format!(
            "{}: {}",
            scan.table,
            if scan.needs_repair {
                "repair-needed"
            } else {
                "ok"
            }
        ));
        lines.push(format!(
            "{}_content_rows: {}",
            scan.table,
            scan.content_rows
                .map(|n| n.to_string())
                .unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "{}_fts_rows: {}",
            scan.table,
            scan.fts_rows
                .map(|n| n.to_string())
                .unwrap_or_else(|| "None".to_string())
        ));
    }
    lines.push("note: read-only scan only — no FTS tables were repaired".to_string());
    if needs_repair {
        lines.push(
            "note: use `/lcm doctor repair apply` to create a backup and repair FTS indexes"
                .to_string(),
        );
    }
    lines.join("\n")
}

/// `/lcm doctor repair apply` — backup-first forced FTS repair (`_doctor_repair_apply_text`,
/// `LCM:command.py:724-763`). The port never degrades to LIKE-only search, so the Python
/// `*_degraded` lines are omitted rather than always-false.
fn doctor_repair_apply_text(engine: &LcmContextEngine) -> String {
    let db = engine
        .config
        .db_path()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let backup = match engine.backup_database() {
        Ok(b) => b,
        Err(e) => {
            return [
                "LCM doctor repair apply".to_string(),
                "status: error".to_string(),
                format!("database_path: {db}"),
                format!("error: backup failed: {e}"),
                "note: repair apply aborted before any FTS tables were repaired".to_string(),
            ]
            .join("\n");
        }
    };
    match engine.store.repair_fts(true) {
        Ok(results) => {
            let mut lines = vec![
                "LCM doctor repair apply".to_string(),
                "status: ok".to_string(),
                format!("database_path: {db}"),
                format!("backup_path: {}", backup.path.display()),
                format!("backup_size: {}", fmt_size(backup.size)),
            ];
            for r in &results {
                lines.push(format!(
                    "{}_rebuilt: {}",
                    r.table,
                    if r.rebuilt { "yes" } else { "no" }
                ));
                lines.push(format!(
                    "{}_triggers_recreated: {}",
                    r.table,
                    if r.triggers_recreated { "yes" } else { "no" }
                ));
            }
            lines.push("note: backup created before repair apply".to_string());
            lines.join("\n")
        }
        Err(e) => [
            "LCM doctor repair apply".to_string(),
            "status: error".to_string(),
            format!("database_path: {db}"),
            format!("backup_path: {}", backup.path.display()),
            format!("backup_size: {}", fmt_size(backup.size)),
            format!("error: FTS repair failed: {e}"),
            "note: backup was created before repair apply".to_string(),
        ]
        .join("\n"),
    }
}

/// The `/lcm` operator command surface — the daemon-authoritative port of `hermes-lcm`'s
/// `command.py`, reusing the engine's existing `lcm_status`/`lcm_doctor` handlers (so the
/// drill-down checks back `/lcm doctor`) plus the maintenance verbs: `backup` (timestamped
/// snapshot), `rotate [apply]` (tail-preserving frontier advance, rolling-backup-first), and
/// `doctor repair [apply]` (FTS index scan/rebuild, backup-first).
#[async_trait]
impl CommandProvider for LcmContextEngine {
    fn name(&self) -> &str {
        "lcm"
    }

    fn commands(&self) -> Vec<CommandSpec> {
        command_specs()
    }

    async fn run_command(
        &self,
        invocation: &CommandInvocation,
        _cx: &CommandCx<'_>,
    ) -> std::result::Result<CommandOutput, CommandError> {
        let tokens: Vec<String> = invocation
            .tokens()
            .iter()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        let sub = tokens.first().map(String::as_str).unwrap_or("status");
        let arg = |i: usize| tokens.get(i).map(String::as_str);
        match sub {
            "" | "status" => {
                let json = self.call_tool("lcm_status", Value::Null).await;
                Ok(CommandOutput::text(pretty_json(&json)))
            }
            "doctor" => match (arg(1), arg(2)) {
                (None, _) => {
                    let json = self.call_tool("lcm_doctor", Value::Null).await;
                    Ok(CommandOutput::text(pretty_json(&json)))
                }
                (Some("repair"), None) => Ok(CommandOutput::text(doctor_repair_text(self))),
                (Some("repair"), Some("apply")) => {
                    Ok(CommandOutput::text(doctor_repair_apply_text(self)))
                }
                (Some("source"), None) => Ok(CommandOutput::text(doctor_source_text(self))),
                (Some("source"), Some("apply")) => {
                    Ok(CommandOutput::text(doctor_source_apply_text(self)))
                }
                _ => Err(CommandError::BadArgs(
                    "unknown /lcm doctor subcommand (try `doctor`, `doctor repair [apply]`, or \
                     `doctor source [apply]`)"
                        .to_string(),
                )),
            },
            "preset" => {
                let json = self.call_tool("lcm_status", Value::Null).await;
                let parsed: Value = serde_json::from_str(&json).unwrap_or(Value::Null);
                match parsed.get("preset_suggestion") {
                    Some(p) if !p.is_null() => {
                        Ok(CommandOutput::text(format!("suggested preset: {p}")))
                    }
                    _ => Ok(CommandOutput::text(
                        "no preset suggestion for the current model window",
                    )),
                }
            }
            "backup" => Ok(CommandOutput::text(backup_text(self))),
            "rotate" => match arg(1) {
                None => Ok(CommandOutput::text(rotate_text(self))),
                Some("apply") => Ok(CommandOutput::text(rotate_apply_text(self))),
                Some(other) => Err(CommandError::BadArgs(format!(
                    "unknown /lcm rotate argument: {other} (try `rotate` or `rotate apply`)"
                ))),
            },
            other => Err(CommandError::BadArgs(format!(
                "unknown /lcm subcommand: {other} (try status|doctor|preset|backup|rotate)"
            ))),
        }
    }
}

/// The static `/lcm` command catalog — the single source for the node command registry (the
/// binary's per-session wrapper advertises these without a live engine instance) and the
/// instance-level [`CommandProvider::commands`].
pub fn command_specs() -> Vec<CommandSpec> {
    vec![CommandSpec::new("lcm")
        .summary("Lossless context management: status, health, preset, backup, rotate")
        .category("Context")
        .args_hint("<status|doctor [repair|source [apply]]|preset|backup|rotate [apply]>")
        .subcommands(["status", "doctor", "preset", "backup", "rotate"])]
}

/// Pretty-print a tool JSON string for human display; fall back to the raw string if it does not
/// parse (the handlers always return JSON, but a defensive fallback keeps output legible).
fn pretty_json(s: &str) -> String {
    match serde_json::from_str::<Value>(s) {
        Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| s.to_string()),
        Err(_) => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::UsageDelta;
    use daemon_core::conversation::{AssistantMsg, ToolResult, ToolTurn};
    use daemon_core::provider::{Capabilities, ModelOutput, Request, ToolCallFormat};
    use daemon_core::{
        Engine, EventSink, Failure, ScriptedProvider, Snapshot, SystemPrompt, ToolCall,
        ToolRegistry, Turn, TurnControl, TurnOutcome,
    };
    use daemon_protocol::{
        HostRequest, HostRequestHandler, HostResponse, HostResponseBody, UserMsg,
    };

    fn aux_with(summary: &str) -> Arc<dyn Provider> {
        Arc::new(ScriptedProvider::new(Vec::new(), summary.to_string()))
    }

    fn model() -> ModelInfo {
        ModelInfo {
            model: "gpt-4o-mini".into(),
            max_context: Some(1000),
        }
    }

    fn convo(n: usize) -> Conversation {
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..n {
            c.push_user(UserMsg::new(format!("message number {i} ").repeat(20)));
            c.push_assistant(AssistantMsg::text(format!("reply number {i} ").repeat(20)));
        }
        c
    }

    #[tokio::test]
    async fn compaction_summarizes_and_assembles_summary_plus_tail() {
        let lcm =
            LcmContextEngine::open_in_memory(aux_with("a terse summary of the past")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let c = convo(50); // 100 turns; fresh tail keeps 32
        let before = c.turns.len();
        let compacted = lcm.compact(c, 100).await;
        // A leading synthetic summary turn + at most fresh_tail_count tail turns.
        assert!(compacted.turns.len() < before, "compaction shrank the body");
        assert!(compacted.turns.len() <= 1 + lcm.config.fresh_tail_count);
        match &compacted.turns[0] {
            Turn::Assistant(a) => assert!(a.text.starts_with(crate::compaction::SUMMARY_SENTINEL)),
            other => panic!("expected leading summary turn, got {other:?}"),
        }
        // A D0 node was persisted and messages were ingested for it.
        assert!(lcm.store().summary_count("s1").unwrap() >= 1);
        assert!(lcm.store().message_count("s1").unwrap() > 0);
    }

    /// A first compaction leaves the system prompt untouched (the prefix-cache regression: the
    /// old behavior appended the LCM note to `Conversation.system`, busting the cached prefix
    /// mid-session).
    #[tokio::test]
    async fn first_compaction_leaves_system_prefix_untouched() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let c = convo(50);
        let system_before = c.system.clone();
        let compacted = lcm.compact(c, 100).await;
        assert!(compacted.turns.len() < 100, "compaction ran");
        assert_eq!(
            compacted.system.text.as_bytes(),
            system_before.text.as_bytes(),
            "compaction must never mutate the system prompt"
        );
    }

    /// The LCM tooling note is a static guidance contribution available from session start —
    /// composed into the system prompt once, before any compaction has run.
    #[tokio::test]
    async fn lcm_guidance_block_present_from_start() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("summary")).unwrap();
        let note = lcm.guidance_block().expect("a static guidance note");
        for tool in ["lcm_grep", "lcm_describe", "lcm_expand"] {
            assert!(note.contains(tool), "note keeps the {tool} guidance");
        }
        // Stable across calls (cache-stable by construction).
        assert_eq!(lcm.guidance_block(), Some(note));
    }

    #[tokio::test]
    async fn tool_pairs_survive_compaction() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        // A tool turn near the end (inside the fresh tail) must survive intact.
        for i in 0..40 {
            c.push_user(UserMsg::new(format!("u{i} ").repeat(30)));
            c.push_assistant(AssistantMsg::text(format!("a{i} ").repeat(30)));
        }
        c.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling tool"),
            calls: vec![(
                ToolCall {
                    call_id: "c1".into(),
                    name: "fs_read".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "c1".into(),
                    ok: true,
                    content: "result body".into(),
                },
            )],
        });
        let compacted = lcm.compact(c, 100).await;
        let last = compacted.turns.last().expect("a tail turn");
        match last {
            Turn::Tool(t) => {
                assert_eq!(t.calls.len(), 1);
                assert_eq!(t.calls[0].0.call_id, t.calls[0].1.call_id);
            }
            other => panic!("expected the tool turn intact at the tail, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recompaction_is_idempotent_noop() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let compacted = lcm.compact(convo(50), 100).await;
        let nodes_after_first = lcm.store().summary_count("s1").unwrap();
        let len_after_first = compacted.turns.len();
        // Re-compacting the already-compacted body adds nothing (region is inside the fresh tail).
        let again = lcm.compact(compacted, 100).await;
        assert_eq!(again.turns.len(), len_after_first, "no further shrink");
        assert_eq!(
            lcm.store().summary_count("s1").unwrap(),
            nodes_after_first,
            "no new summary node"
        );
    }

    #[tokio::test]
    async fn before_turn_mirrors_full_transcript_without_compaction() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        lcm.on_model(&model()); // threshold ~350 tokens; this short convo stays under it
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..3 {
            c.push_user(UserMsg::new(format!("u{i}")));
            c.push_assistant(AssistantMsg::text(format!("a{i}")));
        }
        let pressure = lcm.before_turn(&mut c, None);
        assert!(!pressure.over_budget(), "short convo is under threshold");
        // Every turn was ingested even though no compaction happened.
        assert_eq!(lcm.store().message_count("s1").unwrap(), 6);
        // A second before_turn with no new turns does not duplicate.
        lcm.before_turn(&mut c, None);
        assert_eq!(lcm.store().message_count("s1").unwrap(), 6);
    }

    #[tokio::test]
    async fn compaction_keeps_fresh_tail_byte_equal() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let c = convo(50); // 100 turns; fresh tail keeps 32 from index 68
        let original_tail = c.turns[68..].to_vec();
        let compacted = lcm.compact(c, 100).await;
        // turns[0] is the summary scaffold; the rest must equal the original fresh tail verbatim.
        assert_eq!(&compacted.turns[1..], original_tail.as_slice());
    }

    #[tokio::test]
    async fn rehydration_reconcile_does_not_duplicate_tail() {
        let dir = std::env::temp_dir().join(format!("lcm-reconcile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        // Incarnation 1: compact a long conversation, then close.
        let compacted = {
            let lcm = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("s1"),
                aux_with("s"),
            )
            .unwrap();
            lcm.on_model(&model());
            let out = lcm.compact(convo(50), 100).await;
            out
        };
        let count1 = {
            let reader = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("probe"),
                aux_with("s"),
            )
            .unwrap();
            reader.store().message_count("s1").unwrap()
        };
        // Incarnation 2: rehydrate from the compacted snapshot and run before_turn -> reconcile.
        let mut compacted = compacted;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        lcm2.before_turn(&mut compacted, None);
        let count2 = lcm2.store().message_count("s1").unwrap();
        assert_eq!(
            count2, count1,
            "reconcile rebuilt the tail without duplication"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a fresh, unique data dir for a restart-reconcile test.
    fn reconcile_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lcm-reconcile-{}-{}-{name}",
            std::process::id(),
            RECONCILE_DIR_SEQ.fetch_add(1, Ordering::Relaxed),
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    static RECONCILE_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

    // parity: engine.py::test_existing_session_restart_reconciles_cursor_before_ingest (tests/test_lcm_engine.py:1264)
    #[tokio::test]
    async fn restart_full_transcript_replay_persists_only_new_tail() {
        let dir = reconcile_dir("full-transcript");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        // Incarnation 1: persist a short transcript (no compaction).
        {
            let lcm = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("s1"),
                aux_with("s"),
            )
            .unwrap();
            lcm.on_model(&model());
            let mut c = Conversation::new(SystemPrompt::new("You are concise."));
            c.push_user(UserMsg::new("question before restart"));
            c.push_assistant(AssistantMsg::text("answer before restart"));
            lcm.before_turn(&mut c, None);
        }
        // Incarnation 2: replay the full transcript plus a genuinely-new tool turn.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new("You are concise."));
        replay.push_user(UserMsg::new("question before restart"));
        replay.push_assistant(AssistantMsg::text("answer before restart"));
        replay.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling terminal"),
            calls: vec![(
                ToolCall {
                    call_id: "call_1".into(),
                    name: "terminal".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "call_1".into(),
                    ok: true,
                    content: "terminal output after restart".into(),
                },
            )],
        });
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let roles: Vec<&str> = rows.iter().map(|r| r.role.as_str()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "assistant", "tool"],
            "replay of the durable transcript must not duplicate persisted rows"
        );
        assert_eq!(
            rows.last().unwrap().content.as_deref(),
            Some("terminal output after restart")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Persist `turns` into a fresh incarnation of `session` under `cfg`, then drop it (closing the
    /// store). No compaction runs (`before_turn` only ingests). Returns nothing; the durable rows
    /// remain for the next incarnation.
    async fn persist_durable_turns(
        cfg: &LcmConfig,
        session: &str,
        build: impl FnOnce(&mut Conversation),
    ) {
        let lcm = LcmContextEngine::open_for_session(
            cfg.clone(),
            &SessionId::new(session),
            aux_with("s"),
        )
        .unwrap();
        lcm.on_model(&model());
        let mut c = Conversation::new(SystemPrompt::new("You are concise."));
        build(&mut c);
        lcm.before_turn(&mut c, None);
    }

    // parity: engine.py::test_existing_session_restart_persists_delta_message_matching_store_tail (tests/test_lcm_engine.py:1688)
    #[tokio::test]
    async fn restart_delta_matching_store_tail_is_preserved() {
        let dir = reconcile_dir("delta-matching-tail");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("initial question"));
            c.push_assistant(AssistantMsg::text("initial answer"));
            c.push_user(UserMsg::new("retry"));
        })
        .await;
        // Restart: the gateway hands only the newly-arrived delta, which happens to repeat the
        // durable tail ("retry"). It must be preserved as a genuinely-new row, not treated as replay.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut delta = Conversation::new(SystemPrompt::new("You are concise."));
        delta.push_user(UserMsg::new("retry"));
        lcm2.before_turn(&mut delta, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(
            contents,
            vec!["initial question", "initial answer", "retry", "retry"],
            "an ambiguous delta repeating the tail must append, not wipe the durable transcript"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_persists_single_delta_message_matching_store_tail_with_followup (tests/test_lcm_engine.py:1762)
    #[tokio::test]
    async fn restart_single_delta_matching_tail_with_followup_is_preserved() {
        let dir = reconcile_dir("single-delta-followup");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("retry"));
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut delta = Conversation::new(SystemPrompt::new("You are concise."));
        delta.push_user(UserMsg::new("retry"));
        delta.push_assistant(AssistantMsg::text("next answer"));
        lcm2.before_turn(&mut delta, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(contents, vec!["retry", "retry", "next answer"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_does_not_skip_repeated_non_tail_messages (tests/test_lcm_engine.py:1553)
    #[tokio::test]
    async fn restart_does_not_skip_repeated_non_tail_messages() {
        let dir = reconcile_dir("repeated-non-tail");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("repeatable request"));
            c.push_assistant(AssistantMsg::text("repeatable answer"));
            for i in 0..120 {
                c.push_user(UserMsg::new(format!("tail message before restart {i}")));
            }
        })
        .await;
        let durable = 122;
        // Restart with a short stale replay that repeats an EARLY (non-tail) pair.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "You are concise.\n\n[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        replay.push_user(UserMsg::new("repeatable request"));
        replay.push_assistant(AssistantMsg::text("repeatable answer"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(
            rows.len() as i64,
            durable + 2,
            "repeated non-tail pair appended without wiping the durable transcript"
        );
        assert_eq!(
            rows[rows.len() - 2].content.as_deref(),
            Some("repeatable request")
        );
        assert_eq!(
            rows[rows.len() - 1].content.as_deref(),
            Some("repeatable answer")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_skips_stale_short_no_overlap_snapshot (tests/test_lcm_engine.py:2133)
    #[tokio::test]
    async fn restart_skips_stale_short_no_overlap_snapshot() {
        let dir = reconcile_dir("stale-short-no-overlap");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("old startup question"));
            c.push_assistant(AssistantMsg::text("old startup answer"));
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable tail message {i}")));
            }
        })
        .await;
        let durable = 82;
        // A restarted gateway hands a stale, short snapshot from the START of the session with no
        // overlap with the durable tail. Appending it would duplicate; it must be skipped.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut stale = Conversation::new(SystemPrompt::new("You are concise."));
        stale.push_user(UserMsg::new("old startup question"));
        stale.push_assistant(AssistantMsg::text("old startup answer"));
        lcm2.before_turn(&mut stale, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(
            rows.len() as i64,
            durable,
            "a stale short no-overlap snapshot is skipped, not appended or wiped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_persists_one_message_no_overlap_delta (tests/test_lcm_engine.py:2240)
    #[tokio::test]
    async fn restart_persists_one_message_no_overlap_delta() {
        let dir = reconcile_dir("one-message-no-overlap");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable message {i}")));
            }
        })
        .await;
        let durable = 80;
        // A single, genuinely-new standalone delta with no overlap must be appended (ambiguous
        // singletons are preserved, not skipped).
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut delta = Conversation::new(SystemPrompt::new("You are concise."));
        delta.push_user(UserMsg::new("legitimate standalone delta"));
        lcm2.before_turn(&mut delta, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len() as i64, durable + 1);
        assert_eq!(
            rows.last().unwrap().content.as_deref(),
            Some("legitimate standalone delta")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_scaffold_prefix_does_not_skip_unrelated_new_rows (tests/test_lcm_engine.py:2282)
    #[tokio::test]
    async fn restart_scaffold_prefix_does_not_skip_unrelated_new_rows() {
        let dir = reconcile_dir("scaffold-prefix-unrelated");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable message {i}")));
            }
        })
        .await;
        let durable = 80;
        // Replay begins with a scaffold summary turn, then unrelated NEW rows. The scaffold prefix
        // is skipped, but the new rows must be appended to the durable transcript.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "You are concise.\n\n[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        replay.push_assistant(AssistantMsg::text(
            "[Recent Summary (d0, node 12)]\nEarlier details.\n[Expand for details: hint-12]",
        ));
        replay.push_user(UserMsg::new("unrelated new request"));
        replay.push_assistant(AssistantMsg::text("unrelated new answer"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len() as i64, durable + 2);
        assert_eq!(
            rows[rows.len() - 2].content.as_deref(),
            Some("unrelated new request")
        );
        assert_eq!(
            rows[rows.len() - 1].content.as_deref(),
            Some("unrelated new answer")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_compacted_session_restart_skips_synthetic_context_but_persists_new_tool (tests/test_lcm_engine.py:1315)
    #[tokio::test]
    async fn compacted_restart_skips_synthetic_context_but_persists_new_tool() {
        let dir = reconcile_dir("compacted-new-tool");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        // Incarnation 1: compact a long conversation (produces the synthetic summary scaffold).
        let compacted = {
            let lcm = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("s1"),
                aux_with("s"),
            )
            .unwrap();
            lcm.on_model(&model());
            lcm.compact(convo(50), 100).await
        };
        let count1 = {
            let probe = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("probe"),
                aux_with("s"),
            )
            .unwrap();
            probe.store().message_count("s1").unwrap()
        };
        // Incarnation 2: replay the compacted snapshot plus a genuinely-new tool turn.
        let mut replay = compacted;
        replay.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling terminal"),
            calls: vec![(
                ToolCall {
                    call_id: "call_2".into(),
                    name: "terminal".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "call_2".into(),
                    ok: true,
                    content: "tool output after compacted restart".into(),
                },
            )],
        });
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(
            rows.len() as i64,
            count1 + 2,
            "the synthetic summary context is skipped; only the new tool turn is persisted"
        );
        assert_eq!(
            rows.last().unwrap().content.as_deref(),
            Some("tool output after compacted restart")
        );
        assert_eq!(rows.last().unwrap().tool_call_id.as_deref(), Some("call_2"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_reconciles_full_replay_without_system_prompt (tests/test_lcm_engine.py:1606)
    #[tokio::test]
    async fn restart_full_replay_without_system_anchor_appends_only_new_row() {
        let dir = reconcile_dir("full-replay-no-anchor");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("first question"));
            c.push_assistant(AssistantMsg::text("first answer"));
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new("You are concise."));
        replay.push_user(UserMsg::new("first question"));
        replay.push_assistant(AssistantMsg::text("first answer"));
        replay.push_user(UserMsg::new("new question after restart"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(
            contents,
            vec![
                "first question",
                "first answer",
                "new question after restart"
            ],
            "a multi-row full replay is proof enough without a system anchor"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_reconciles_complete_replay_without_system_prompt (tests/test_lcm_engine.py:1651)
    #[tokio::test]
    async fn restart_complete_replay_without_new_rows_is_noop() {
        let dir = reconcile_dir("complete-replay-noop");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("first question"));
            c.push_assistant(AssistantMsg::text("first answer"));
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new("You are concise."));
        replay.push_user(UserMsg::new("first question"));
        replay.push_assistant(AssistantMsg::text("first answer"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(contents, vec!["first question", "first answer"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_persists_scaffolded_delta_message_matching_store_tail (tests/test_lcm_engine.py:1800)
    #[tokio::test]
    async fn restart_scaffolded_delta_matching_tail_is_preserved() {
        let dir = reconcile_dir("scaffolded-delta");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("initial question"));
            c.push_assistant(AssistantMsg::text("initial answer"));
            c.push_user(UserMsg::new("retry"));
        })
        .await;
        // Restart with the LCM system note and a delta repeating only the durable tail row.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "You are concise.\n\n[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        replay.push_user(UserMsg::new("retry"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(
            contents,
            vec!["initial question", "initial answer", "retry", "retry"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_persists_scaffolded_delta_message_matching_store_tail_with_followup (tests/test_lcm_engine.py:1845)
    #[tokio::test]
    async fn restart_scaffolded_delta_with_followup_is_preserved() {
        let dir = reconcile_dir("scaffolded-delta-followup");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("initial question"));
            c.push_assistant(AssistantMsg::text("initial answer"));
            c.push_user(UserMsg::new("retry"));
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "You are concise.\n\n[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        replay.push_user(UserMsg::new("retry"));
        replay.push_assistant(AssistantMsg::text("next answer"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(
            contents,
            vec![
                "initial question",
                "initial answer",
                "retry",
                "retry",
                "next answer"
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_restart_reconciliation_filtered_singleton_tail_stays_ambiguous (tests/test_lcm_engine.py:2397)
    #[tokio::test]
    async fn restart_filtered_singleton_tail_stays_ambiguous() {
        let dir = reconcile_dir("filtered-singleton");
        let base = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        // Incarnation 1 has no ignore filter: both rows are persisted.
        persist_durable_turns(&base, "s1", |c| {
            c.push_user(UserMsg::new("Cronjob Response: heartbeat"));
            c.push_user(UserMsg::new("real singleton tail"));
        })
        .await;
        // Incarnation 2 enables the ignore filter; the singleton delta repeats the visible tail
        // but stays ambiguous and must be preserved.
        let cfg2 = LcmConfig {
            ignore_message_patterns: vec!["^Cronjob Response:".to_string()],
            ..base.clone()
        };
        let lcm2 =
            LcmContextEngine::open_for_session(cfg2, &SessionId::new("s1"), aux_with("s")).unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new("You are concise."));
        replay.push_user(UserMsg::new("real singleton tail"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        let contents: Vec<&str> = rows.iter().filter_map(|r| r.content.as_deref()).collect();
        assert_eq!(
            contents,
            vec![
                "Cronjob Response: heartbeat",
                "real singleton tail",
                "real singleton tail"
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_large_session_restart_reconciles_beyond_short_tail_window (tests/test_lcm_engine.py:1510)
    #[tokio::test]
    async fn restart_large_session_reconciles_beyond_short_tail_window() {
        let dir = reconcile_dir("large-session");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            for i in 0..5000 {
                c.push_user(UserMsg::new(format!("message before restart {i}")));
            }
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new("You are concise."));
        for i in 0..5000 {
            replay.push_user(UserMsg::new(format!("message before restart {i}")));
        }
        replay.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling terminal"),
            calls: vec![(
                ToolCall {
                    call_id: "call_large".into(),
                    name: "terminal".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "call_large".into(),
                    ok: true,
                    content: "large-session tool output after restart".into(),
                },
            )],
        });
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len(), 5002, "5000 replayed rows + assistant + tool");
        assert_eq!(rows.last().unwrap().role, "tool");
        assert_eq!(
            rows.last().unwrap().tool_call_id.as_deref(),
            Some("call_large")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_persists_repeated_prefix_after_scaffold_only_prefix (tests/test_lcm_engine.py:2338)
    #[tokio::test]
    async fn restart_repeated_head_after_scaffold_only_prefix_is_preserved() {
        let dir = reconcile_dir("scaffold-repeated-head");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("old first question"));
            c.push_assistant(AssistantMsg::text("old first answer"));
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable tail after scaffold {i}")));
            }
        })
        .await;
        let durable = 82;
        // A stale replay: scaffold summary turn, then the durable HEAD pair (not the tail). The
        // scaffold-only prefix is skipped; the repeated pair must be appended (no stale proof —
        // the LCM-note system prompt is not a plain anchor).
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        replay.push_assistant(AssistantMsg::text(
            "[Recent Summary (d0, node 12)]\nEarlier details.\n[Expand for details: hint-12]",
        ));
        replay.push_user(UserMsg::new("old first question"));
        replay.push_assistant(AssistantMsg::text("old first answer"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len() as i64, durable + 2);
        assert_eq!(
            rows[rows.len() - 2].content.as_deref(),
            Some("old first question")
        );
        assert_eq!(
            rows[rows.len() - 1].content.as_deref(),
            Some("old first answer")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_persists_cleanup_sensitive_scaffolded_repeated_tail (tests/test_lcm_engine.py:1891)
    #[tokio::test]
    async fn restart_literal_json_assistant_tail_delta_is_preserved() {
        let dir = reconcile_dir("literal-json-tail");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        let literal_json = r#"[{"text": "visible literal JSON payload", "type": "thinking"}]"#;
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("older question"));
            c.push_assistant(AssistantMsg::text("older answer"));
            c.push_user(UserMsg::new("retry"));
            c.push_assistant(AssistantMsg::text(literal_json));
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "You are concise.\n\n[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        replay.push_user(UserMsg::new("retry"));
        replay.push_assistant(AssistantMsg::text(literal_json));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len(), 6);
        let tail: Vec<&str> = rows[2..]
            .iter()
            .filter_map(|r| r.content.as_deref())
            .collect();
        assert_eq!(tail, vec!["retry", literal_json, "retry", literal_json]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_skips_exact_lcm_system_scaffold (tests/test_lcm_engine.py:2095)
    #[tokio::test]
    async fn restart_system_only_replay_leaves_store_untouched() {
        let dir = reconcile_dir("system-only");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new("tail before restart"));
        })
        .await;
        // Rust keeps the system prompt off the turn stream, so an "exact LCM system scaffold"
        // replay is a conversation with zero turns; nothing must be ingested or deleted.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new(
            "You are concise.\n\n[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]",
        ));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content.as_deref(), Some("tail before restart"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ingest_redacts_sensitive_content_when_enabled() {
        let cfg = LcmConfig {
            sensitive_patterns_enabled: true,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new(
            "here is my api_key=ABCDEF0123456789 please keep it", // gitleaks:allow (test fixture)
        ));
        lcm.before_turn(&mut c, None);
        let rows = lcm.store().session_messages("s1").unwrap();
        let body = rows[0].content.as_deref().unwrap();
        assert!(body.contains("name=api_key"), "secret redacted: {body}");
        assert!(!body.contains("ABCDEF0123456789"));
    }

    #[tokio::test]
    async fn before_turn_redacts_provider_facing_conversation_in_place() {
        let cfg = LcmConfig {
            sensitive_patterns_enabled: true,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new(
            "here is my api_key=ABCDEF0123456789 please keep it", // gitleaks:allow (test fixture)
        ));
        c.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling"),
            calls: vec![(
                ToolCall {
                    call_id: "c1".into(),
                    name: "t".into(),
                    args: r#"{"api_key": "ABCDEF0123456789"}"#.into(), // gitleaks:allow
                },
                ToolResult {
                    call_id: "c1".into(),
                    ok: true,
                    content: "password = hunter2hunter2".into(), // gitleaks:allow
                },
            )],
        });
        lcm.before_turn(&mut c, None);
        // The provider-facing conversation itself was sanitized, not just the store write.
        let user_text = match &c.turns[0] {
            Turn::User(u) => u.text.as_str(),
            other => panic!("unexpected turn: {other:?}"),
        };
        assert!(
            user_text.contains("[LCM sensitive redaction:"),
            "user text: {user_text}"
        );
        assert!(!user_text.contains("ABCDEF0123456789"));
        let Turn::Tool(t) = &c.turns[1] else {
            panic!("expected tool turn");
        };
        assert!(
            !t.calls[0].0.args.contains("ABCDEF0123456789"),
            "args redacted: {}",
            t.calls[0].0.args
        );
        assert!(
            !t.calls[0].1.content.contains("hunter2hunter2"),
            "result redacted: {}",
            t.calls[0].1.content
        );
    }

    #[tokio::test]
    async fn before_turn_quarantines_runaway_assistant_output_from_replay() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new("go"));
        // Degenerate loop output: one short line repeated far past the 64 KiB quarantine floor.
        let runaway = "loop detected heartbeat ping\n".repeat(4000);
        c.push_assistant(AssistantMsg::text(runaway.clone()));
        lcm.before_turn(&mut c, None);
        let Turn::Assistant(a) = &c.turns[1] else {
            panic!("expected assistant turn");
        };
        assert_ne!(a.text, runaway, "runaway output replaced");
        assert!(
            a.text.contains("placeholder"),
            "quarantine placeholder in replay: {}",
            &a.text[..a.text.len().min(120)]
        );
        // No externalization dir on an in-memory bank -> the volatile placeholder.
        assert!(a.text.starts_with("[LCM active replay placeholder:"));
    }

    #[tokio::test]
    async fn ingest_storage_guard_externalizes_and_gc_rewrites() {
        let dir = std::env::temp_dir().join(format!("lcm-gc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            large_output_transcript_gc_enabled: true,
            large_output_externalization_enabled: true,
            ..LcmConfig::default()
        };
        let lcm = LcmContextEngine::open_for_session(
            cfg.clone(),
            &SessionId::new("s1"),
            aux_with("summary"),
        )
        .unwrap();
        lcm.on_model(&model());
        let big_b64 = "QUJDREVG".repeat(700); // > 4096 base64 chars
        let big_text = "verbose tool output line ".repeat(600); // > 12k chars threshold
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        // Early tool turns (outside the fresh tail): one large base64 run embedded in prose (the
        // always-on storage guard spills just the run), one whole-body oversized tool output (the
        // opt-in threshold path spills the entire result).
        c.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling tool"),
            calls: vec![(
                ToolCall {
                    call_id: "c1".into(),
                    name: "fs_read".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "c1".into(),
                    ok: true,
                    content: format!("payload: {big_b64}"),
                },
            )],
        });
        c.push_tool(ToolTurn {
            assistant: AssistantMsg::text("calling tool again"),
            calls: vec![(
                ToolCall {
                    call_id: "c2".into(),
                    name: "fs_read".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "c2".into(),
                    ok: true,
                    content: big_text.clone(),
                },
            )],
        });
        for i in 0..50 {
            c.push_user(UserMsg::new(format!("message number {i} ").repeat(20)));
            c.push_assistant(AssistantMsg::text(format!("reply number {i} ").repeat(20)));
        }
        let _ = lcm.compact(c, 100).await;

        let rows = lcm.store().session_messages("s1").unwrap();
        let row_for = |id: &str| {
            rows.iter()
                .find(|r| r.role == "tool" && r.tool_call_id.as_deref() == Some(id))
                .unwrap_or_else(|| panic!("tool row {id}"))
        };
        let ext_dir = cfg.externalization_dir().unwrap();
        let recover = |body: &str| {
            let reference = crate::externalize::extract_ref(body).expect("a ref");
            crate::externalize::read_externalized(ext_dir.as_path(), &reference).unwrap()
        };

        // The whole-body tool output (kind=tool_output) is summarized then GC-rewritten in place,
        // with the row's cached token estimate following the compact placeholder.
        let gc_row = row_for("c2");
        let gc_body = gc_row.content.as_deref().unwrap();
        assert!(
            gc_body.starts_with("[GC'd externalized tool output:"),
            "GC'd: {gc_body}"
        );
        assert!(gc_row.token_estimate < 50, "estimate follows the rewrite");
        assert_eq!(recover(gc_body), big_text, "recoverable after GC");

        // The base64 run was only a slice of its row (`payload: ` prose remains inline), so the
        // GC kind-guard must leave the ingest placeholder untouched
        // (`_maybe_gc_compacted_tool_results`, `LCM:engine.py:3459-3469`).
        let guard_row = row_for("c1");
        let guard_body = guard_row.content.as_deref().unwrap();
        assert!(
            guard_body.starts_with("payload: [Externalized LCM ingest payload:"),
            "partial spill kept inline: {guard_body}"
        );
        assert!(!guard_body.contains(&big_b64), "payload bytes left the row");
        assert_eq!(
            recover(guard_body),
            big_b64,
            "the externalized run is the base64 payload itself"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ignored_session_skips_all_store_writes() {
        let cfg = LcmConfig {
            ignore_session_patterns: vec!["s1".to_string()],
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = convo(50); // would normally compact
        let pressure = lcm.before_turn(&mut c, None);
        assert!(
            !pressure.over_budget(),
            "ignored session reports no budget pressure"
        );
        let compacted = lcm.compact(c.clone(), 100).await;
        assert_eq!(
            compacted.turns.len(),
            c.turns.len(),
            "no compaction for ignored session"
        );
        assert_eq!(lcm.store().message_count("s1").unwrap(), 0, "no ingest");
        assert_eq!(
            lcm.store().summary_count("s1").unwrap(),
            0,
            "no summary nodes"
        );
    }

    #[tokio::test]
    async fn stateless_session_is_read_only() {
        let cfg = LcmConfig {
            stateless_session_patterns: vec!["scratch-*".to_string()],
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("scratch-1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new("hello"));
        lcm.before_turn(&mut c, None);
        assert_eq!(
            lcm.store().message_count("scratch-1").unwrap(),
            0,
            "stateless = no writes"
        );
    }

    #[tokio::test]
    async fn ignore_message_pattern_drops_matching_turns() {
        let cfg = LcmConfig {
            ignore_message_patterns: vec![r"(?i)^/debug\b".to_string()],
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new("/debug dump the state"));
        c.push_user(UserMsg::new("a real substantive question"));
        lcm.before_turn(&mut c, None);
        let rows = lcm.store().session_messages("s1").unwrap();
        assert_eq!(rows.len(), 1, "the /debug turn was filtered");
        assert!(rows[0]
            .content
            .as_deref()
            .unwrap()
            .contains("real substantive"));
    }

    #[tokio::test]
    async fn leaf_floor_gate_suppresses_pressure_for_small_backlog() {
        // Default leaf_chunk_tokens (20k) dwarfs this ~4k-token conversation: over the 350-token
        // threshold, but compact() could not fill one leaf chunk -> no advertised pressure
        // (`_leaf_compaction_candidate_status`).
        let lcm = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = convo(50);
        let pressure = lcm.before_turn(&mut c, None);
        assert!(pressure.used_tokens > 350, "well over the threshold");
        assert!(!pressure.over_budget(), "leaf floor suppresses pressure");
    }

    #[tokio::test]
    async fn pressure_reported_when_backlog_meets_leaf_floor() {
        let cfg = LcmConfig {
            leaf_chunk_tokens: 100, // floor below the backlog so eligibility holds
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = convo(50);
        let pressure = lcm.before_turn(&mut c, None);
        assert!(pressure.over_budget(), "eligible backlog reports pressure");
    }

    #[tokio::test]
    async fn after_response_records_provider_usage_for_status() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        lcm.after_response(&UsageDelta {
            input_tokens: 900,
            output_tokens: 120,
            api_calls: 1,
            cache_read_tokens: 450,
            cache_write_tokens: 30,
            reasoning_tokens: 64,
            cost_micros: 0,
        });
        let status: serde_json::Value =
            serde_json::from_str(&lcm.call_tool("lcm_status", serde_json::json!({})).await)
                .unwrap();
        assert_eq!(status["last_prompt_tokens"], 900, "provider report wins");
        assert_eq!(status["last_input_tokens"], 900);
        assert_eq!(status["last_output_tokens"], 120);
        assert_eq!(status["last_cache_read_tokens"], 450);
        assert_eq!(status["last_cache_write_tokens"], 30);
        assert_eq!(status["last_reasoning_tokens"], 64);
        assert_eq!(status["cache_metrics_available"], true);
        assert_eq!(status["cache_read_ratio"], 0.5);
    }

    #[tokio::test]
    async fn session_end_flushes_final_turns_and_finalizes_lifecycle() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new("only turn"));
        // No before_turn ran — on_session_end alone must persist the tail and finalize.
        lcm.on_session_end(&SessionId::new("s1"), &c);
        assert_eq!(lcm.store().message_count("s1").unwrap(), 1, "final flush");
        let row = lcm.store().get_lifecycle("s1").unwrap().unwrap();
        assert_eq!(row.current_session_id, None, "current cleared");
        assert_eq!(row.last_finalized_session_id.as_deref(), Some("s1"));
        assert!(row.last_finalized_at.is_some());
    }

    #[tokio::test]
    async fn rollover_prunes_retained_dag_and_carries_over_summaries() {
        let cfg = LcmConfig {
            new_session_retain_depth: 0, // `/new` drops everything at depth < ... i.e. all nodes
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let compacted = lcm.compact(convo(50), 100).await;
        assert!(lcm.store().summary_count("s1").unwrap() >= 1);
        // Drive the §10 trait hook (not the inherent variant) so the seam itself is exercised.
        ContextEngine::rollover_session(
            &lcm,
            &SessionId::new("s1"),
            &SessionId::new("s2"),
            Some(&compacted),
            true,
        );
        // retain_depth=0 leaves nothing to carry over.
        assert_eq!(lcm.store().summary_count("s1").unwrap(), 0);
        assert_eq!(lcm.store().summary_count("s2").unwrap(), 0);
        let row = lcm.store().get_lifecycle("s1").unwrap().unwrap();
        assert_eq!(row.last_finalized_session_id.as_deref(), Some("s1"));
        assert!(row.last_reset_at.is_some(), "reset stamped");
        // The engine is now bound to s2.
        let bound = lcm.store().get_lifecycle("s2").unwrap().unwrap();
        assert_eq!(bound.current_session_id.as_deref(), Some("s2"));
    }

    #[tokio::test]
    async fn rollover_with_retained_depth_carries_nodes_into_new_session() {
        // Default retain depth is 2, which keeps nothing from a single D0 pass; retain everything
        // instead so the carry-over path is observable.
        let cfg = LcmConfig {
            new_session_retain_depth: -1,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let compacted = lcm.compact(convo(50), 100).await;
        let nodes_before = lcm.store().summary_count("s1").unwrap();
        assert!(nodes_before >= 1);
        let moved = lcm.rollover_sessions("s1", "s2", Some(&compacted), true);
        assert_eq!(moved as i64, nodes_before, "all retained nodes moved");
        assert_eq!(lcm.store().summary_count("s1").unwrap(), 0);
        assert_eq!(lcm.store().summary_count("s2").unwrap(), nodes_before);
    }

    /// The §10 `on_session_reset` trait hook (fired by the engine on a full-clear rewind) prunes
    /// the bound session's retained DAG per `new_session_retain_depth` and stamps the lifecycle
    /// reset — the `/new` semantics without a host-must-remember inherent call.
    #[tokio::test]
    async fn trait_reset_hook_prunes_retained_dag_and_stamps_lifecycle() {
        let cfg = LcmConfig {
            new_session_retain_depth: 0, // drop every retained node on reset
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let _ = lcm.compact(convo(50), 100).await;
        assert!(lcm.store().summary_count("s1").unwrap() >= 1);

        ContextEngine::on_session_reset(&lcm, &SessionId::new("s1"));

        assert_eq!(lcm.store().summary_count("s1").unwrap(), 0, "DAG pruned");
        let row = lcm.store().get_lifecycle("s1").unwrap().unwrap();
        assert!(row.last_reset_at.is_some(), "reset stamped");
    }

    /// A reset addressed to a session this per-session instance is not bound to is a wiring bug —
    /// it must be skipped (never prune another session's DAG).
    #[tokio::test]
    async fn trait_reset_hook_skips_mismatched_session() {
        let cfg = LcmConfig {
            new_session_retain_depth: 0,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let _ = lcm.compact(convo(50), 100).await;
        let nodes = lcm.store().summary_count("s1").unwrap();
        assert!(nodes >= 1);

        ContextEngine::on_session_reset(&lcm, &SessionId::new("someone-else"));

        assert_eq!(
            lcm.store().summary_count("s1").unwrap(),
            nodes,
            "mismatched reset touched nothing"
        );
    }

    #[tokio::test]
    async fn empty_lifecycle_rows_are_pruned_at_bind_over_threshold() {
        let cfg = LcmConfig {
            empty_lifecycle_gc_threshold: 3,
            empty_lifecycle_gc_max_age_hours: None, // prune regardless of age (test env)
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        // Simulate gateway-restart orphans: lifecycle rows that never ingested anything.
        for i in 0..4 {
            lcm.store()
                .bind_session(&format!("orphan-{i}"), &format!("orphan-{i}"), 1.0)
                .unwrap();
        }
        assert_eq!(lcm.store().lifecycle_row_count().unwrap(), 4);
        lcm.on_session_start(&SessionId::new("live"));
        // 4 orphans + live row exceeded 3 -> the empty orphans were pruned, live survives.
        assert_eq!(lcm.store().lifecycle_row_count().unwrap(), 1);
        assert!(lcm.store().get_lifecycle("live").unwrap().is_some());
    }

    #[tokio::test]
    async fn status_surfaces_preset_and_filter_state() {
        let cfg = LcmConfig {
            sensitive_patterns_enabled: true,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model()); // max_context 1000 -> no preset
        lcm.on_session_start(&SessionId::new("s1"));
        let status: serde_json::Value =
            serde_json::from_str(&lcm.call_tool("lcm_status", serde_json::json!({})).await)
                .unwrap();
        assert_eq!(
            status["ingest_protection"]["sensitive_patterns_enabled"],
            true
        );
        assert_eq!(status["session_filters"]["ignored"], false);
        assert_eq!(status["context_length"], 1000);
        assert!(
            status["preset_suggestion"]["suggested_preset"].is_null(),
            "1000-token window has no preset"
        );
        assert_eq!(status["last_compression_status"], "idle");
    }

    #[tokio::test]
    async fn per_session_instances_attribute_summaries_correctly() {
        let dir = std::env::temp_dir().join(format!("lcm-attr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        let s1 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        let s2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s2"), aux_with("s"))
                .unwrap();
        s1.on_model(&model());
        s2.on_model(&model());
        let (_r1, _r2) = tokio::join!(s1.compact(convo(50), 100), s2.compact(convo(40), 100));

        let reader =
            LcmContextEngine::open_for_session(cfg, &SessionId::new("reader"), aux_with("s"))
                .unwrap();
        assert_eq!(
            reader.store().summary_count("s1").unwrap(),
            1,
            "s1 attributed"
        );
        assert_eq!(
            reader.store().summary_count("s2").unwrap(),
            1,
            "s2 attributed"
        );
        assert_eq!(
            reader.store().summary_count("reader").unwrap(),
            0,
            "no cross-attribution"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// M8 (spec §15): concurrent sessions over **one shared bank** ingest and attribute both their
    /// summary nodes and raw `messages` to the correct `session_id`, with no cross-contamination.
    /// Strengthens `per_session_instances_attribute_summaries_correctly` by also driving the
    /// `before_turn` ingest path and asserting raw-message attribution is disjoint by session.
    #[tokio::test]
    async fn m8_concurrent_sessions_isolate_ingest_and_attribution() {
        let dir = std::env::temp_dir().join(format!("lcm-m8-attr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        let a = LcmContextEngine::open_for_session(
            cfg.clone(),
            &SessionId::new("alpha"),
            aux_with("s"),
        )
        .unwrap();
        let b =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("beta"), aux_with("s"))
                .unwrap();
        a.on_model(&model());
        b.on_model(&model());

        // Ingest each session's live transcript, then compact both concurrently over the one bank.
        let (mut ca, mut cb) = (convo(50), convo(40));
        a.before_turn(&mut ca, None);
        b.before_turn(&mut cb, None);
        let (_ra, _rb) = tokio::join!(a.compact(ca, 100), b.compact(cb, 100));

        let reader =
            LcmContextEngine::open_for_session(cfg, &SessionId::new("reader"), aux_with("s"))
                .unwrap();
        let store = reader.store();
        // Summary nodes attributed to the right session, none leaked to a third.
        assert_eq!(store.summary_count("alpha").unwrap(), 1, "alpha attributed");
        assert_eq!(store.summary_count("beta").unwrap(), 1, "beta attributed");
        assert_eq!(
            store.summary_count("reader").unwrap(),
            0,
            "no cross-attribution"
        );
        // Raw messages ingested under each session are disjoint and non-empty.
        assert!(store.message_count("alpha").unwrap() > 0, "alpha ingested");
        assert!(store.message_count("beta").unwrap() > 0, "beta ingested");
        assert_eq!(
            store.message_count("reader").unwrap(),
            0,
            "reader ingested nothing"
        );
        // Every persisted node carries the session id it belongs to.
        for sid in ["alpha", "beta"] {
            for node in store.get_session_nodes(sid, None, 100).unwrap() {
                assert_eq!(node.session_id, sid, "node {} mis-attributed", node.node_id);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A host that approves everything (the §8 recovery loop never gates in this test).
    struct NoopHost;

    #[async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved {
                    approved: true,
                    allow_permanent: false,
                    reason: None,
                },
            }
        }
    }

    /// A model provider that fails its first call with `ContextOverflow` (forcing the §8 ->
    /// §10 compact-and-retry-once path) then succeeds. `max_context` is large so the *pre-turn*
    /// budget check never compacts — the only compaction is the error-driven one under test.
    struct OverflowOnceProvider {
        calls: AtomicU64,
    }

    #[async_trait]
    impl Provider for OverflowOnceProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(200_000),
            }
        }

        async fn chat(&self, _req: Request) -> std::result::Result<ModelOutput, Failure> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed);
            if n == 0 {
                Err(Failure::ContextOverflow(
                    "prompt exceeds the model window".into(),
                ))
            } else {
                Ok(ModelOutput {
                    text: "done after compaction".into(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                    usage: UsageDelta::default(),
                    ..Default::default()
                })
            }
        }
    }

    /// M8 (spec §15): end-to-end context-overflow recovery through the real `Engine` driving a real
    /// `LcmContextEngine`. The first model call overflows; the engine compacts via LCM exactly once
    /// and retries; the retry succeeds. Asserts the turn completed, the model was called exactly
    /// twice (one compact-then-retry), and LCM actually compacted (a summary node was written).
    #[tokio::test]
    async fn m8_context_overflow_compacts_via_lcm_and_retries_once() {
        let aux = aux_with("a compacted summary of the earlier conversation");
        let lcm = Arc::new(LcmContextEngine::open_in_memory(aux).unwrap());
        let provider = Arc::new(OverflowOnceProvider {
            calls: AtomicU64::new(0),
        });

        // Seed a long conversation (100 turns) so the forced compaction has a region beyond the
        // fresh tail to summarize.
        let mut snapshot = Snapshot::fresh(SessionId::new("overflow"));
        snapshot.conversation = convo(50);
        let mut engine =
            Engine::from_snapshot(snapshot, provider.clone(), Arc::new(ToolRegistry::new()))
                .with_context_engine(lcm.clone());
        engine.push_user(UserMsg::new("continue please"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .expect("turn completes after a single compact + retry");

        assert!(
            matches!(outcome, TurnOutcome::Completed(_)),
            "turn completed"
        );
        assert_eq!(
            provider.calls.load(Ordering::Relaxed),
            2,
            "the overflow drove exactly one compact-then-retry"
        );
        assert!(
            lcm.store().summary_count("overflow").unwrap() >= 1,
            "LCM compacted the overflowed conversation (a summary node was written)"
        );
    }

    /// An assembly cap (`max_assembly_tokens`) drives force overflow end to end: `before_turn`
    /// advertises the cap as the compaction target even under the polite threshold, and
    /// `compact` gets the assembled context under the cap.
    #[tokio::test]
    async fn assembly_cap_forces_overflow_compaction_under_the_cap() {
        let cfg = LcmConfig {
            max_assembly_tokens: 300,
            fresh_tail_count: 4,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("a terse summary")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = convo(10); // ~1900 tokens, far over the 300-token cap
        let p = lcm.before_turn(&mut c, None);
        assert_eq!(
            p.budget_tokens,
            Some(299),
            "cap-driven force overflow targets the cap, not the polite threshold"
        );
        let out = lcm.compact(c, 299).await;
        let tok = Tokenizer::for_model("gpt-4o-mini");
        assert!(
            tok.count_conversation(&out) <= 300,
            "assembled context fits the cap (got {})",
            tok.count_conversation(&out)
        );
        assert!(
            lcm.store().summary_count("s1").unwrap() >= 1,
            "force overflow summarized the whole eligible region"
        );
        let status: serde_json::Value =
            serde_json::from_str(&lcm.call_tool("lcm_status", serde_json::json!({})).await)
                .unwrap();
        assert_eq!(status["last_compression_status"], "compacted");
    }

    /// When everything sits inside the fresh tail (nothing to summarize) but the context is over
    /// the cap, compaction still recovers by reassembling under the cap without a new summary
    /// node — Python's `overflow_recovery` status.
    #[tokio::test]
    async fn assembly_cap_overflow_recovery_drops_tail_without_a_new_node() {
        let cfg = LcmConfig {
            max_assembly_tokens: 300,
            // Default fresh_tail_count (32) keeps every turn of a short conversation in-tail.
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..6 {
            c.push_assistant(AssistantMsg::text(format!("chatter {i} ").repeat(30)));
        }
        let before = c.turns.len();
        let p = lcm.before_turn(&mut c, None);
        assert!(p.budget_tokens.is_some(), "cap pressure advertised");
        let out = lcm.compact(c, 299).await;
        let tok = Tokenizer::for_model("gpt-4o-mini");
        assert!(tok.count_conversation(&out) <= 300, "recovered under cap");
        assert!(out.turns.len() < before, "over-cap tail turns were dropped");
        assert_eq!(
            lcm.store().summary_count("s1").unwrap(),
            0,
            "no new summary node on the recovery path"
        );
        let status: serde_json::Value =
            serde_json::from_str(&lcm.call_tool("lcm_status", serde_json::json!({})).await)
                .unwrap();
        assert_eq!(status["last_compression_status"], "overflow_recovery");
    }

    /// Deferred maintenance: a conversation carrying raw-backlog debt advertises a catch-up
    /// compaction even while under the polite threshold (`should_compress` preflight,
    /// `LCM:engine.py:693-694`).
    #[tokio::test]
    async fn deferred_maintenance_debt_advertises_catchup_pressure_under_threshold() {
        let cfg = LcmConfig {
            deferred_maintenance_enabled: true,
            leaf_chunk_tokens: 10,
            fresh_tail_count: 2,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model()); // threshold = 350 tokens
        lcm.on_session_start(&SessionId::new("s1"));
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..6 {
            c.push_user(UserMsg::new(format!("small turn {i}")));
        }
        // Well under the 350-token threshold, but the 4-turn backlog outside the tail exceeds the
        // 10-token leaf chunk: the preflight records debt and advertises a catch-up target.
        let p = lcm.before_turn(&mut c, None);
        assert!(p.used_tokens < 350, "conversation is under the threshold");
        assert_eq!(
            p.budget_tokens,
            Some(p.used_tokens - 1),
            "debt-carrying turn advertises a catch-up compaction"
        );
        assert!(p.over_budget(), "catch-up target triggers a compact call");
        // Without the feature the same shape reports no pressure.
        let quiet = LcmContextEngine::open(
            LcmConfig {
                leaf_chunk_tokens: 10,
                fresh_tail_count: 2,
                ..LcmConfig::in_memory()
            },
            aux_with("s"),
        )
        .unwrap();
        quiet.on_model(&model());
        quiet.on_session_start(&SessionId::new("s1"));
        let mut c2 = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..6 {
            c2.push_user(UserMsg::new(format!("small turn {i}")));
        }
        assert!(
            !quiet.before_turn(&mut c2, None).over_budget(),
            "no debt, under threshold: nothing triggers a compact call"
        );
    }

    // ---- /lcm operator surface (backup / rotate / doctor repair) -------------------------------

    async fn run_lcm(engine: &LcmContextEngine, args: &str) -> String {
        engine
            .run_command(
                &CommandInvocation {
                    name: "lcm".into(),
                    args: args.into(),
                    session: None,
                },
                &CommandCx::node(),
            )
            .await
            .expect("command ok")
            .text
    }

    // parity: command.py::test_doctor_source_reports_legacy_blank_rows (tests/test_lcm_command.py:440)
    #[tokio::test]
    async fn doctor_source_scans_legacy_blank_rows() {
        let (lcm, dir) = durable_engine("doctor-source-scan", 32);
        let mut c = convo(2); // 4 attributed rows via the normal write path
        lcm.before_turn(&mut c, None);
        lcm.store()
            .insert_legacy_blank_source_row("s1", "user", "legacy blank row")
            .unwrap();
        let text = run_lcm(&lcm, "doctor source").await;
        assert!(text.contains("status: normalization-needed"), "{text}");
        assert!(text.contains("legacy_blank_messages: 1"), "{text}");
        assert!(text.contains("would_update_messages: 1"), "{text}");
        assert!(text.contains("target_source: unknown"), "{text}");
        assert!(
            text.contains("no source rows were updated"),
            "read-only: {text}"
        );
        // Read-only scan did not mutate the blank row.
        assert_eq!(
            lcm.store()
                .source_stats(None)
                .unwrap()
                .legacy_blank_source_messages,
            1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: command.py::test_doctor_source_apply_normalizes_legacy_blank_rows (tests/test_lcm_command.py:451)
    #[tokio::test]
    async fn doctor_source_apply_normalizes_legacy_blank_rows() {
        let (lcm, dir) = durable_engine("doctor-source-apply", 32);
        let mut c = convo(2);
        lcm.before_turn(&mut c, None);
        lcm.store()
            .insert_legacy_blank_source_row("s1", "user", "legacy blank row")
            .unwrap();
        let text = run_lcm(&lcm, "doctor source apply").await;
        assert!(text.contains("status: ok"), "{text}");
        assert!(text.contains("updated_messages: 1"), "{text}");
        assert!(text.contains("legacy_blank_before: 1"), "{text}");
        assert!(text.contains("legacy_blank_after: 0"), "{text}");
        assert!(text.contains("backup_path: "), "backup-first: {text}");
        // The blank row is now attributed to the explicit unknown bucket.
        let stats = lcm.store().source_stats(None).unwrap();
        assert_eq!(stats.legacy_blank_source_messages, 0);
        // A second scan reports nothing to do.
        let rescan = run_lcm(&lcm, "doctor source").await;
        assert!(rescan.contains("status: ok"), "{rescan}");
        assert!(rescan.contains("would_update_messages: 0"), "{rescan}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_session_restart_skips_stale_short_snapshot_with_externalized_head_payload (tests/test_lcm_engine.py:2185)
    #[tokio::test]
    async fn restart_stale_snapshot_with_externalized_head_payload_is_skipped() {
        let dir = reconcile_dir("stale-externalized-head");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            large_output_externalization_enabled: true,
            ..LcmConfig::default()
        };
        let data_uri = format!("data:image/png;base64,{}", "QUJDREVGRw==".repeat(500));
        let head_user = format!("old startup image {data_uri}");
        persist_durable_turns(&cfg, "s1", |c| {
            c.push_user(UserMsg::new(head_user.clone()));
            c.push_assistant(AssistantMsg::text("old startup answer"));
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable tail message {i}")));
            }
        })
        .await;
        // The head row was externalized at the write boundary.
        {
            let probe = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("probe"),
                aux_with("s"),
            )
            .unwrap();
            let rows = probe.store().session_messages("s1").unwrap();
            assert_eq!(rows.len(), 82);
            let head = rows[0].content.as_deref().unwrap();
            assert!(
                head.contains("[Externalized LCM ingest payload:"),
                "head externalized: {head}"
            );
            assert!(!head.contains(&data_uri));
        }
        // Restart with a stale short snapshot re-supplying the raw head (inline payload intact).
        // Identity restoration must recognize it as the externalized durable head and skip it.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut stale = Conversation::new(SystemPrompt::new("You are concise."));
        stale.push_user(UserMsg::new(head_user.clone()));
        stale.push_assistant(AssistantMsg::text("old startup answer"));
        lcm2.before_turn(&mut stale, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(
            rows.len(),
            82,
            "the stale externalized-head snapshot is skipped, not duplicated"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_restart_reconciliation_filtered_prefix_does_not_create_stale_proof (tests/test_lcm_engine.py:2487)
    #[tokio::test]
    async fn restart_filtered_prefix_does_not_create_stale_proof() {
        let dir = reconcile_dir("filtered-prefix-stale");
        let base = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&base, "s1", |c| {
            c.push_user(UserMsg::new("Cronjob Response: heartbeat"));
            c.push_user(UserMsg::new("real prefix question"));
            c.push_assistant(AssistantMsg::text("real prefix answer"));
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable tail after filter {i}")));
            }
        })
        .await;
        let durable = 83;
        // Incarnation 2 enables the ignore filter. The delta repeats what is now the *visible*
        // durable head — but the stale-snapshot proof must use the RAW durable prefix (the cron
        // row included), so this batch stays ambiguous and is preserved.
        let cfg2 = LcmConfig {
            ignore_message_patterns: vec!["^Cronjob Response:".to_string()],
            ..base.clone()
        };
        let lcm2 =
            LcmContextEngine::open_for_session(cfg2, &SessionId::new("s1"), aux_with("s")).unwrap();
        lcm2.on_model(&model());
        let mut replay = Conversation::new(SystemPrompt::new("You are concise."));
        replay.push_user(UserMsg::new("real prefix question"));
        replay.push_assistant(AssistantMsg::text("real prefix answer"));
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(
            rows.len() as i64,
            durable + 2,
            "filtered history must not create stale-replay proof"
        );
        assert_eq!(
            rows[rows.len() - 2].content.as_deref(),
            Some("real prefix question")
        );
        assert_eq!(
            rows[rows.len() - 1].content.as_deref(),
            Some("real prefix answer")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_existing_compacted_session_restart_ignores_preserved_objective_anchor (tests/test_lcm_engine.py:1375)
    #[tokio::test]
    async fn restart_ignores_preserved_objective_anchor() {
        let dir = reconcile_dir("preserved-objective");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            fresh_tail_count: 2,
            ..LcmConfig::default()
        };
        let latest_request = "increase kanban autonomy";
        // Incarnation 1: the newest real user turn is followed by two tool turns, so with a
        // 2-turn fresh tail the compaction preserves it as an objective anchor inside the
        // synthetic summary scaffold.
        let compacted = {
            let lcm = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("s1"),
                aux_with("Older board cleanup summary."),
            )
            .unwrap();
            lcm.on_model(&model());
            let mut c = Conversation::new(SystemPrompt::new("You are concise."));
            for i in 0..20 {
                c.push_user(UserMsg::new(format!("clean up boards {i} ").repeat(20)));
                c.push_assistant(AssistantMsg::text(format!("inspecting {i} ").repeat(20)));
            }
            c.push_user(UserMsg::new(latest_request));
            for call in ["call_1", "call_2"] {
                c.push_tool(ToolTurn {
                    assistant: AssistantMsg::text(format!("inspect via {call}")),
                    calls: vec![(
                        ToolCall {
                            call_id: call.into(),
                            name: "inspect".into(),
                            args: "{}".into(),
                        },
                        ToolResult {
                            call_id: call.into(),
                            ok: true,
                            content: format!("{call} output"),
                        },
                    )],
                });
            }
            lcm.compact(c, 100).await
        };
        // The compacted scaffold carries the preserved-objective anchor.
        let scaffold_text = match &compacted.turns[0] {
            Turn::Assistant(a) => a.text.clone(),
            other => panic!("expected scaffold, got {other:?}"),
        };
        assert!(
            scaffold_text.contains("Current user objective preserved"),
            "anchor emitted: {scaffold_text}"
        );
        let count1 = {
            let probe = LcmContextEngine::open_for_session(
                cfg.clone(),
                &SessionId::new("probe"),
                aux_with("s"),
            )
            .unwrap();
            probe.store().message_count("s1").unwrap()
        };
        // Incarnation 2: replay the compacted snapshot plus one new user message.
        let mut replay = compacted;
        replay.push_user(UserMsg::new("follow-up after restart"));
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        lcm2.before_turn(&mut replay, None);
        let rows = lcm2.store().session_messages("s1").unwrap();
        assert_eq!(rows.len() as i64, count1 + 1, "only the follow-up is new");
        assert_eq!(
            rows.last().unwrap().content.as_deref(),
            Some("follow-up after restart")
        );
        let objective_repeats = rows
            .iter()
            .filter(|r| r.content.as_deref() == Some(latest_request))
            .count();
        assert_eq!(
            objective_repeats, 1,
            "the anchored objective is not re-ingested"
        );
        assert!(
            rows.iter().all(|r| !r
                .content
                .as_deref()
                .unwrap_or("")
                .contains("Current user objective preserved")),
            "the scaffold anchor never lands in the store"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // parity: engine.py::test_lcm_status_reports_ingest_reconciliation_diagnostics (tests/test_lcm_engine.py:2542)
    #[tokio::test]
    async fn status_reports_ingest_reconciliation_diagnostics() {
        let dir = reconcile_dir("status-reconcile");
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            ..LcmConfig::default()
        };
        persist_durable_turns(&cfg, "s1", |c| {
            for i in 0..80 {
                c.push_user(UserMsg::new(format!("durable message {i}")));
            }
        })
        .await;
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        let mut delta = Conversation::new(SystemPrompt::new("You are concise."));
        delta.push_user(UserMsg::new("status delta"));
        lcm2.before_turn(&mut delta, None);
        let payload: Value =
            serde_json::from_str(&lcm2.call_tool("lcm_status", Value::Null).await).unwrap();
        assert_eq!(
            payload["ingest_reconciliation"]["reason"], "persisted ambiguous delta",
            "status: {payload}"
        );
        assert_eq!(
            payload["ingest_reconciliation"]["action"],
            "persisted batch"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An engine-level compaction fixture: 6 alternating user/assistant turns of ~43 tokens each
    /// (the Python `"Message {i}: " + "chunk " * 35` shape), so with `fresh_tail_count = 2` the
    /// eligible region is turns 0..4.
    fn chunky_convo(words_per_turn: usize) -> Conversation {
        let mut c = Conversation::new(SystemPrompt::new("You are a helpful assistant."));
        for i in 0..6 {
            let body = format!("Message {i}: {}", "chunk ".repeat(words_per_turn));
            if i % 2 == 0 {
                c.push_user(UserMsg::new(body));
            } else {
                c.push_assistant(AssistantMsg::text(body));
            }
        }
        c
    }

    /// Seed a pre-existing summary node (the Python `engine._dag.add_node(SummaryNode(...))`
    /// fixture) at `depth`.
    fn seed_node(lcm: &LcmContextEngine, session: &str, depth: i64, summary: &str, age: f64) {
        lcm.store()
            .add_node(&crate::store::NewNode {
                session_id: session.to_string(),
                depth,
                summary: summary.to_string(),
                token_count: 40,
                source_token_count: 80,
                source_ids: Vec::new(),
                source_type: if depth == 0 {
                    crate::store::SourceType::Messages
                } else {
                    crate::store::SourceType::Nodes
                },
                created_at: 1_000.0 - age,
                earliest_at: None,
                latest_at: None,
                expand_hint: format!("Expand for details about: {summary}"),
            })
            .unwrap();
    }

    // parity: engine.py::test_dynamic_leaf_chunk_sizing_compacts_only_oldest_bounded_raw_chunk (tests/test_lcm_engine.py:4029)
    #[tokio::test]
    async fn dynamic_leaf_chunk_compacts_only_oldest_bounded_chunk() {
        let cfg = LcmConfig {
            fresh_tail_count: 2,
            leaf_chunk_tokens: 50,
            dynamic_leaf_chunk_enabled: true,
            dynamic_leaf_chunk_max: 120,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(
            cfg,
            aux_with("Dynamic leaf summary.\nExpand for details about: oldest raw chunk"),
        )
        .unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let c = chunky_convo(35);
        let region_texts: Vec<String> = c.turns[..4]
            .iter()
            .map(|t| match t {
                Turn::User(u) => u.text.clone(),
                Turn::Assistant(a) => a.text.clone(),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        let compacted = lcm.compact(c, 100_000).await;
        // Exactly one D0 node, covering ONLY the oldest two region rows (the bounded chunk).
        assert_eq!(lcm.store().summary_count("s1").unwrap(), 1);
        let nodes = lcm.store().get_session_nodes("s1", Some(0), 10).unwrap();
        assert_eq!(nodes[0].source_ids, vec![1, 2], "oldest bounded chunk only");
        // The un-summarized region remainder and the tail stay in the compacted context.
        let texts: Vec<String> = compacted
            .turns
            .iter()
            .map(|t| match t {
                Turn::User(u) => u.text.clone(),
                Turn::Assistant(a) => a.text.clone(),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert!(texts.contains(&region_texts[2]), "Message 2 kept raw");
        assert!(texts.contains(&region_texts[3]), "Message 3 kept raw");
        assert!(
            !texts.iter().any(|t| t == &region_texts[0]),
            "Message 0 summarized"
        );
        // The durable transcript is intact (all 6 rows).
        assert_eq!(lcm.store().message_count("s1").unwrap(), 6);
    }

    /// An aux provider that fails retry-worthily whenever the prompt covers the second turn
    /// (`"Message 1:"`), succeeding on smaller chunks — the engine-level analog of the Python
    /// `flaky_summary` that raises "context length exceeded" for large sources.
    struct MarkerFailAux {
        calls: AtomicU64,
    }

    #[async_trait]
    impl Provider for MarkerFailAux {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: false,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(200_000),
            }
        }

        async fn chat(&self, req: Request) -> std::result::Result<ModelOutput, Failure> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let prompt = req
                .messages
                .first()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            if prompt.contains("Message 1:") {
                return Err(Failure::ContextOverflow("context length exceeded".into()));
            }
            Ok(ModelOutput {
                text: "Recovered smaller leaf summary.\nExpand for details about: oldest raw chunk"
                    .into(),
                ..Default::default()
            })
        }
    }

    // parity: engine.py::test_adaptive_leaf_rescue_retries_with_smaller_oldest_chunk (tests/test_lcm_engine.py:4085)
    #[tokio::test]
    async fn adaptive_leaf_rescue_retries_with_smaller_oldest_chunk() {
        let cfg = LcmConfig {
            fresh_tail_count: 2,
            leaf_chunk_tokens: 50,
            dynamic_leaf_chunk_enabled: true,
            dynamic_leaf_chunk_max: 120,
            // The Python mock has no circuit breaker in the loop; keep the single route closed
            // across the two failed levels of attempt 1 so attempt 2 can succeed.
            summary_circuit_breaker_failure_threshold: 10,
            ..LcmConfig::in_memory()
        };
        let aux = Arc::new(MarkerFailAux {
            calls: AtomicU64::new(0),
        });
        let lcm = LcmContextEngine::open(cfg, aux.clone()).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        let compacted = lcm.compact(chunky_convo(35), 100_000).await;
        // Attempt 1 (2-turn chunk) fails L1+L2 retry-worthily; the rescue shrinks to the oldest
        // single turn and attempt 2's L1 succeeds: 3 aux calls in total.
        assert_eq!(
            aux.calls.load(Ordering::Relaxed),
            3,
            "L1+L2 failed, retry L1 succeeded"
        );
        assert_eq!(lcm.store().summary_count("s1").unwrap(), 1);
        let nodes = lcm.store().get_session_nodes("s1", Some(0), 10).unwrap();
        assert_eq!(
            nodes[0].source_ids,
            vec![1],
            "only the rescued single-turn chunk"
        );
        let texts: Vec<&str> = compacted
            .turns
            .iter()
            .map(|t| match t {
                Turn::User(u) => u.text.as_str(),
                Turn::Assistant(a) => a.text.as_str(),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert!(
            texts.iter().any(|t| t.starts_with("Message 1:")),
            "Message 1 kept raw"
        );
        assert!(
            texts.iter().any(|t| t.starts_with("Message 2:")),
            "Message 2 kept raw"
        );
        assert!(
            texts.iter().any(|t| t.starts_with("Message 3:")),
            "Message 3 kept raw"
        );
    }

    // parity: engine.py::test_unlimited_depth_condenses_beyond_ten (tests/test_lcm_engine.py:9029)
    #[tokio::test]
    async fn unlimited_condensation_depth_reaches_d12() {
        let cfg = LcmConfig {
            fresh_tail_count: 2,
            leaf_chunk_tokens: 1,
            condensation_fanin: 2,
            incremental_max_depth: -1,
            ..LcmConfig::in_memory()
        };
        let lcm = LcmContextEngine::open(
            cfg,
            aux_with("Condensed.\nExpand for details about: deep nodes"),
        )
        .unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        for i in 0..3 {
            seed_node(&lcm, "s1", 11, &format!("Deep node {i}"), 10.0 + i as f64);
        }
        lcm.compact(chunky_convo(35), 100_000).await;
        assert!(
            lcm.store().count_at_depth("s1", 12).unwrap() >= 1,
            "max_depth=-1 condenses beyond depth 10"
        );
    }

    fn durable_engine(tag: &str, fresh_tail: usize) -> (LcmContextEngine, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("lcm-op-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            fresh_tail_count: fresh_tail,
            ..LcmConfig::default()
        };
        let lcm = LcmContextEngine::open(cfg, aux_with("s")).unwrap();
        lcm.on_model(&model());
        lcm.on_session_start(&SessionId::new("s1"));
        (lcm, dir)
    }

    #[test]
    fn fmt_size_matches_the_python_precision_ladder() {
        assert_eq!(fmt_size(512), "512 B");
        assert_eq!(fmt_size(2048), "2.00 KB");
        assert_eq!(fmt_size(15 * 1024), "15.0 KB");
        assert_eq!(fmt_size(200 * 1024), "200 KB");
        assert_eq!(fmt_size(5 * 1024 * 1024), "5.00 MB");
    }

    #[tokio::test]
    async fn backup_command_snapshots_a_durable_bank_and_refuses_in_memory() {
        let (lcm, dir) = durable_engine("backup", 32);
        let mut c = convo(2);
        lcm.before_turn(&mut c, None);
        let text = run_lcm(&lcm, "backup").await;
        assert!(text.contains("status: ok"), "{text}");
        let backup_line = text
            .lines()
            .find(|l| l.starts_with("backup_path: "))
            .expect("backup path line");
        let path = std::path::Path::new(&backup_line["backup_path: ".len()..]);
        assert!(path.exists(), "snapshot written");
        assert!(path.starts_with(dir.join("backups").join("lcm")));
        // The snapshot is a readable database with the same session rows.
        let copy = Store::open(path).unwrap();
        assert_eq!(
            copy.message_count("s1").unwrap(),
            lcm.store().message_count("s1").unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);

        let mem = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        let text = run_lcm(&mem, "backup").await;
        assert!(text.contains("status: error"), "{text}");
        assert!(text.contains("database file does not exist"), "{text}");
    }

    #[tokio::test]
    async fn rotate_previews_applies_and_noops_idempotently() {
        let (lcm, dir) = durable_engine("rotate", 2);
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..6 {
            c.push_user(UserMsg::new(format!("turn number {i}")));
        }
        lcm.before_turn(&mut c, None);
        assert_eq!(lcm.store().message_count("s1").unwrap(), 6);

        // Preview: proposes a frontier past the 4 pre-tail rows without persisting anything.
        let text = run_lcm(&lcm, "rotate").await;
        assert!(text.contains("status: preview"), "{text}");
        assert!(text.contains("pre_tail_message_count: 4"), "{text}");
        assert_eq!(lcm.store().get_frontier("s1").unwrap(), 0, "read-only");

        // Apply: rolling backup written, frontier advanced to just below the tail.
        let text = run_lcm(&lcm, "rotate apply").await;
        assert!(text.contains("status: ok"), "{text}");
        let frontier = lcm.store().get_frontier("s1").unwrap();
        assert!(frontier > 0, "frontier advanced");
        let slot = lcm.config.rotate_backup_path().unwrap();
        assert!(slot.exists(), "rolling slot written");
        let first_backup = std::fs::metadata(&slot).unwrap().modified().unwrap();

        // Idempotent rerun: noop, and the previous known-good rolling snapshot is preserved.
        let text = run_lcm(&lcm, "rotate apply").await;
        assert!(text.contains("status: noop"), "{text}");
        assert!(text.contains("frontier_already_ahead"), "{text}");
        assert_eq!(lcm.store().get_frontier("s1").unwrap(), frontier);
        assert_eq!(
            std::fs::metadata(&slot).unwrap().modified().unwrap(),
            first_backup,
            "noop rerun did not overwrite the rolling snapshot"
        );

        // The pre-tail raw rows are still in the store (lossless recovery contract).
        assert_eq!(lcm.store().message_count("s1").unwrap(), 6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn rotate_refuses_without_a_bound_session() {
        let lcm = LcmContextEngine::open_in_memory(aux_with("s")).unwrap();
        let text = run_lcm(&lcm, "rotate").await;
        assert!(text.contains("status: refused"), "{text}");
        assert!(text.contains("reason: no_active_session"), "{text}");
        let text = run_lcm(&lcm, "rotate apply").await;
        assert!(text.contains("status: refused"), "{text}");
        assert!(text.contains("no backup was created"), "{text}");
    }

    #[tokio::test]
    async fn doctor_repair_scans_read_only_and_apply_backs_up_first() {
        let (lcm, dir) = durable_engine("repair", 32);
        let mut c = convo(2);
        lcm.before_turn(&mut c, None);

        let text = run_lcm(&lcm, "doctor repair").await;
        assert!(text.contains("status: ok"), "{text}");
        assert!(text.contains("messages_fts: ok"), "{text}");
        assert!(text.contains("nodes_fts: ok"), "{text}");
        assert!(text.contains("read-only scan only"), "{text}");

        let text = run_lcm(&lcm, "doctor repair apply").await;
        assert!(text.contains("status: ok"), "{text}");
        assert!(text.contains("backup_path: "), "{text}");
        assert!(text.contains("messages_fts_rebuilt: no"), "{text}");
        assert!(text.contains("nodes_fts_rebuilt: no"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
