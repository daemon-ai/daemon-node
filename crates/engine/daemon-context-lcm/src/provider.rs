//! [`LcmContextEngine`] — the `daemon-core` [`ContextEngine`] (§10) backed by the summary DAG.
//!
//! `on_model` sizes the compaction threshold + selects the tokenizer from the model window;
//! `before_turn` measures token [`Pressure`] (with a boundary cooldown after a no-op compaction);
//! `compact` runs the real LCM pass (`compaction::run_compaction`) — summarize the region outside
//! the fresh tail into the DAG and reassemble `[system] + [summary] + [fresh tail]`; the
//! session-lifecycle hooks bind the conversation frontier.

use crate::compaction::{leading_scaffold_count, run_compaction};
use crate::config::LcmConfig;
use crate::error::Result;
use crate::escalation::SummaryCircuitBreaker;
use crate::ingest::flatten_turns;
use crate::patterns::{build_session_match_keys, MessagePatterns, SessionGlobs};
use crate::protection::protect_message_for_ingest;
use crate::store::Store;
use crate::tokens::Tokenizer;
use crate::tools::{ToolCx, TOOL_NAMES};
use async_trait::async_trait;
use daemon_common::SessionId;
use daemon_core::tools::ToolDef;
use daemon_core::{Conversation, ContextEngine, ModelInfo, Pressure, Provider, Turn};
use serde_json::Value;
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
    /// When the last compaction was a no-op (for the boundary cooldown).
    last_noop_at: Option<Instant>,
    /// The number of live conversation turns already ingested into `messages` this incarnation.
    cursor: usize,
    /// Per-turn ingest index: `turn_store_ids[i]` are the `store_id`s persisted for live turn `i`
    /// (empty for a synthetic summary turn). Kept aligned with the live conversation so compaction
    /// can attribute D0 `source_ids` without re-ingesting.
    turn_store_ids: Vec<Vec<i64>>,
    /// Whether this incarnation has reconciled its tail against the durable store yet (once per
    /// incarnation, on the first ingest).
    reconciled: bool,
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
        let store = match config.db_path() {
            Some(path) => Store::open(path)?,
            None => Store::open_in_memory()?,
        };
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
        let _ = engine
            .store
            .bind_session(session.as_str(), session.as_str(), Self::now());
        Ok(engine)
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
        let (session_id, tokenizer, threshold_tokens, context_length, compaction_count, session_ignored, session_stateless) = {
            let state = self.state.lock().expect("lcm state poisoned");
            (
                effective_session(&state.session_id),
                state.tokenizer.clone(),
                state.threshold_tokens,
                state.context_length,
                state.compaction_count,
                state.session_ignored,
                state.session_stateless,
            )
        };
        let cx = ToolCx {
            store: &self.store,
            config: &self.config,
            tokenizer: &tokenizer,
            aux: self.aux.as_ref(),
            session_id: &session_id,
            threshold_tokens,
            context_length,
            compaction_count,
            session_ignored,
            session_stateless,
            ignored_message_count: ignored_message_count(),
        };
        crate::tools::dispatch(&cx, name, args).await
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
        if !state.reconciled {
            let frontier = self.store.get_frontier(&session).unwrap_or(0);
            let _ = self.store.delete_messages_after(&session, frontier);
            state.cursor = 0;
            state.turn_store_ids.clear();
            state.reconciled = true;
        }
        let scaffold = leading_scaffold_count(&conv.turns);
        let tok = state.tokenizer.clone();
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
                    .map(|m| protect_message_for_ingest(m, &self.config, ext_dir.as_deref()))
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
        if let Some(max) = model.max_context {
            state.context_length = Some(max as usize);
            state.threshold_tokens = Some((max as f64 * self.config.context_threshold) as usize);
        }
    }

    fn before_turn(&self, conv: &Conversation, budget: Option<usize>) -> Pressure {
        // Keep the durable transcript current before measuring pressure (so the tools see this turn).
        self.ingest_current(conv);
        let state = self.state.lock().expect("lcm state poisoned");
        let used_tokens = state.tokenizer.count_conversation(conv);
        // Ignored/stateless sessions are never compacted (no store writes) — report no budget so the
        // turn loop never calls `compact` for them (§12.5).
        let filtered = state.session_ignored || state.session_stateless;
        // Boundary cooldown: after a no-op compaction, report no budget for a short window so the
        // engine doesn't re-attempt a compaction it can't make progress on every turn.
        let in_cooldown = matches!(state.last_noop_at, Some(t) if t.elapsed() < BOUNDARY_COOLDOWN);
        let budget_tokens = if in_cooldown || filtered {
            None
        } else {
            budget.or(state.threshold_tokens)
        };
        Pressure {
            used_tokens,
            budget_tokens,
        }
    }

    async fn compact(&self, conv: Conversation, _budget: usize) -> Conversation {
        // Catch up the ingest index to the live conversation (the ReAct loop may have appended turns
        // since `before_turn`), then snapshot the bits compaction needs and run it without holding
        // the state lock across the aux-provider `await`s. The breaker + index are taken out and
        // restored afterwards.
        // Ignored/stateless sessions are read-only: never write summary nodes / advance the frontier.
        {
            let state = self.state.lock().expect("lcm state poisoned");
            if state.session_ignored || state.session_stateless {
                return conv;
            }
        }
        self.ingest_current(&conv);
        let (tokenizer, session_id, mut breakers, first_compaction, index) = {
            let mut state = self.state.lock().expect("lcm state poisoned");
            let breakers = std::mem::take(&mut state.breakers);
            let index = std::mem::take(&mut state.turn_store_ids);
            (
                state.tokenizer.clone(),
                state.session_id.clone(),
                breakers,
                state.compaction_count == 0,
                index,
            )
        };
        let session = effective_session(&session_id);

        let (compacted, did_compact, new_index) = run_compaction(
            &self.store,
            &tokenizer,
            &self.config,
            &self.aux_chain,
            &mut breakers,
            &session,
            first_compaction,
            index,
            conv,
            Self::now(),
        )
        .await;

        let mut state = self.state.lock().expect("lcm state poisoned");
        state.breakers = breakers;
        state.cursor = new_index.len();
        state.turn_store_ids = new_index;
        if did_compact {
            state.compaction_count += 1;
            state.last_noop_at = None;
        } else {
            state.last_noop_at = Some(Instant::now());
        }
        compacted
    }

    fn on_session_start(&self, session: &SessionId) {
        let mut state = self.state.lock().expect("lcm state poisoned");
        state.session_id = session.as_str().to_string();
        self.refresh_session_filters(session.as_str(), &mut state);
        drop(state);
        let _ = self
            .store
            .bind_session(session.as_str(), session.as_str(), Self::now());
    }

    fn on_session_end(&self, session: &SessionId, _conv: &Conversation) {
        let count = self.store.summary_count(session.as_str()).unwrap_or(0);
        tracing::debug!(session = %session, summaries = count, "lcm: session ended");
    }

    /// The advisory names of the `lcm_*` tools this engine owns; the host registers their actual
    /// dispatch through the §12 [`ToolRegistry`](daemon_core::tools) (see [`Self::tool_defs`]).
    fn tools(&self) -> Vec<String> {
        TOOL_NAMES.iter().map(|s| s.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::conversation::{AssistantMsg, ToolResult, ToolTurn};
    use daemon_core::{ScriptedProvider, SystemPrompt, ToolCall, Turn};
    use daemon_protocol::UserMsg;

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
        let lcm = LcmContextEngine::open_in_memory(aux_with("a terse summary of the past")).unwrap();
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
        let pressure = lcm.before_turn(&c, None);
        assert!(!pressure.over_budget(), "short convo is under threshold");
        // Every turn was ingested even though no compaction happened.
        assert_eq!(lcm.store().message_count("s1").unwrap(), 6);
        // A second before_turn with no new turns does not duplicate.
        lcm.before_turn(&c, None);
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
            let lcm =
                LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                    .unwrap();
            lcm.on_model(&model());
            let out = lcm.compact(convo(50), 100).await;
            out
        };
        let count1 = {
            let reader =
                LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("probe"), aux_with("s"))
                    .unwrap();
            reader.store().message_count("s1").unwrap()
        };
        // Incarnation 2: rehydrate from the compacted snapshot and run before_turn -> reconcile.
        let lcm2 =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
                .unwrap();
        lcm2.on_model(&model());
        lcm2.before_turn(&compacted, None);
        let count2 = lcm2.store().message_count("s1").unwrap();
        assert_eq!(count2, count1, "reconcile rebuilt the tail without duplication");
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
        c.push_user(UserMsg::new("here is my api_key=ABCDEF0123456789 please keep it"));
        lcm.before_turn(&c, None);
        let rows = lcm.store().session_messages("s1").unwrap();
        let body = rows[0].content.as_deref().unwrap();
        assert!(body.contains("name=api_key"), "secret redacted: {body}");
        assert!(!body.contains("ABCDEF0123456789"));
    }

    #[tokio::test]
    async fn ingest_storage_guard_externalizes_and_gc_rewrites() {
        let dir = std::env::temp_dir().join(format!("lcm-gc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = LcmConfig {
            data_dir: dir.clone(),
            bank: "default".to_string(),
            large_output_transcript_gc_enabled: true,
            ..LcmConfig::default()
        };
        let lcm =
            LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("summary"))
                .unwrap();
        lcm.on_model(&model());
        let big_b64 = "QUJDREVG".repeat(700); // > 4096 base64 chars
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        // An early tool turn (outside the fresh tail) carrying a large base64 payload.
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
        for i in 0..50 {
            c.push_user(UserMsg::new(format!("message number {i} ").repeat(20)));
            c.push_assistant(AssistantMsg::text(format!("reply number {i} ").repeat(20)));
        }
        let _ = lcm.compact(c, 100).await;

        // The tool-result row is summarized + externalized, then GC-rewritten in place.
        let rows = lcm.store().session_messages("s1").unwrap();
        let tool_row = rows
            .iter()
            .find(|r| r.role == "tool")
            .expect("the tool result row");
        let body = tool_row.content.as_deref().unwrap();
        assert!(body.starts_with("[GC'd externalized tool output:"), "GC'd: {body}");
        assert!(!body.contains(&big_b64), "payload bytes left the row");

        // The original payload is recoverable from disk via the ref.
        let reference = crate::externalize::extract_ref(body).expect("a ref");
        let recovered = crate::externalize::read_externalized(
            cfg.externalization_dir().unwrap().as_path(),
            &reference,
        )
        .unwrap();
        assert_eq!(recovered, big_b64, "the externalized run is the base64 payload itself");
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
        let c = convo(50); // would normally compact
        let pressure = lcm.before_turn(&c, None);
        assert!(!pressure.over_budget(), "ignored session reports no budget pressure");
        let compacted = lcm.compact(c.clone(), 100).await;
        assert_eq!(compacted.turns.len(), c.turns.len(), "no compaction for ignored session");
        assert_eq!(lcm.store().message_count("s1").unwrap(), 0, "no ingest");
        assert_eq!(lcm.store().summary_count("s1").unwrap(), 0, "no summary nodes");
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
        lcm.before_turn(&c, None);
        assert_eq!(lcm.store().message_count("scratch-1").unwrap(), 0, "stateless = no writes");
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
        lcm.before_turn(&c, None);
        let rows = lcm.store().session_messages("s1").unwrap();
        assert_eq!(rows.len(), 1, "the /debug turn was filtered");
        assert!(rows[0].content.as_deref().unwrap().contains("real substantive"));
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
            serde_json::from_str(&lcm.call_tool("lcm_status", serde_json::json!({})).await).unwrap();
        assert_eq!(status["protection"]["sensitive_patterns_enabled"], true);
        assert_eq!(status["filters"]["session_ignored"], false);
        assert_eq!(status["context_length"], 1000);
        assert!(status["preset_suggestion"].is_null(), "1000-token window has no preset");
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
        let s1 = LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s1"), aux_with("s"))
            .unwrap();
        let s2 = LcmContextEngine::open_for_session(cfg.clone(), &SessionId::new("s2"), aux_with("s"))
            .unwrap();
        s1.on_model(&model());
        s2.on_model(&model());
        let (_r1, _r2) = tokio::join!(s1.compact(convo(50), 100), s2.compact(convo(40), 100));

        let reader =
            LcmContextEngine::open_for_session(cfg, &SessionId::new("reader"), aux_with("s")).unwrap();
        assert_eq!(reader.store().summary_count("s1").unwrap(), 1, "s1 attributed");
        assert_eq!(reader.store().summary_count("s2").unwrap(), 1, "s2 attributed");
        assert_eq!(
            reader.store().summary_count("reader").unwrap(),
            0,
            "no cross-attribution"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
