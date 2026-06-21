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
use crate::store::Store;
use crate::tokens::Tokenizer;
use crate::tools::{ToolCx, TOOL_NAMES};
use async_trait::async_trait;
use daemon_common::SessionId;
use daemon_core::tools::ToolDef;
use daemon_core::{Conversation, ContextEngine, ModelInfo, Pressure, Provider};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// After a compaction that could not shrink anything (region already inside the fresh tail), suppress
/// re-triggering for this long so the engine doesn't re-attempt a no-op every turn (§6.3 cooldown).
const BOUNDARY_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(60);

/// Mutable per-session runtime state (small; behind a mutex so the sync hooks can update it).
struct State {
    /// The compaction threshold (model-window-derived), if known.
    threshold_tokens: Option<usize>,
    /// The active session id (keys summary nodes / lifecycle rows).
    session_id: String,
    /// The model-aware token counter (heuristic until `on_model`).
    tokenizer: Tokenizer,
    /// The aux-provider circuit breaker (carried across compactions).
    breaker: SummaryCircuitBreaker,
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

impl Default for State {
    fn default() -> Self {
        Self {
            threshold_tokens: None,
            session_id: String::new(),
            tokenizer: Tokenizer::heuristic(),
            breaker: SummaryCircuitBreaker::new(),
            compaction_count: 0,
            last_noop_at: None,
            cursor: 0,
            turn_store_ids: Vec::new(),
            reconciled: false,
        }
    }
}

/// The LCM context engine over a single summary-store bank.
pub struct LcmContextEngine {
    config: LcmConfig,
    store: Store,
    aux: Arc<dyn Provider>,
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
        Ok(Self {
            config,
            store,
            aux,
            state: Mutex::new(State::default()),
        })
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
        let (session_id, tokenizer, threshold_tokens, compaction_count) = {
            let state = self.state.lock().expect("lcm state poisoned");
            (
                effective_session(&state.session_id),
                state.tokenizer.clone(),
                state.threshold_tokens,
                state.compaction_count,
            )
        };
        let cx = ToolCx {
            store: &self.store,
            config: &self.config,
            tokenizer: &tokenizer,
            aux: self.aux.as_ref(),
            session_id: &session_id,
            threshold_tokens,
            compaction_count,
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
        while state.cursor < conv.turns.len() {
            let idx = state.cursor;
            if idx < scaffold {
                // A synthetic summary scaffold turn carries no real messages.
                state.turn_store_ids.push(Vec::new());
            } else {
                let rows = flatten_turns(std::slice::from_ref(&conv.turns[idx]), &tok);
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
            state.threshold_tokens = Some((max as f64 * self.config.context_threshold) as usize);
        }
    }

    fn before_turn(&self, conv: &Conversation, budget: Option<usize>) -> Pressure {
        // Keep the durable transcript current before measuring pressure (so the tools see this turn).
        self.ingest_current(conv);
        let state = self.state.lock().expect("lcm state poisoned");
        let used_tokens = state.tokenizer.count_conversation(conv);
        // Boundary cooldown: after a no-op compaction, report no budget for a short window so the
        // engine doesn't re-attempt a compaction it can't make progress on every turn.
        let in_cooldown = matches!(state.last_noop_at, Some(t) if t.elapsed() < BOUNDARY_COOLDOWN);
        let budget_tokens = if in_cooldown {
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
        self.ingest_current(&conv);
        let (tokenizer, session_id, mut breaker, first_compaction, index) = {
            let mut state = self.state.lock().expect("lcm state poisoned");
            let breaker = std::mem::take(&mut state.breaker);
            let index = std::mem::take(&mut state.turn_store_ids);
            (
                state.tokenizer.clone(),
                state.session_id.clone(),
                breaker,
                state.compaction_count == 0,
                index,
            )
        };
        let session = effective_session(&session_id);

        let (compacted, did_compact, new_index) = run_compaction(
            &self.store,
            &tokenizer,
            &self.config,
            self.aux.as_ref(),
            &mut breaker,
            &session,
            first_compaction,
            index,
            conv,
            Self::now(),
        )
        .await;

        let mut state = self.state.lock().expect("lcm state poisoned");
        state.breaker = breaker;
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
