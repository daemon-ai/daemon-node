//! [`LcmContextEngine`] — the `daemon-core` [`ContextEngine`] (§10) backed by the summary DAG.
//!
//! `on_model` sizes the compaction threshold + selects the tokenizer from the model window;
//! `before_turn` measures token [`Pressure`] (with a boundary cooldown after a no-op compaction);
//! `compact` runs the real LCM pass (`compaction::run_compaction`) — summarize the region outside
//! the fresh tail into the DAG and reassemble `[system] + [summary] + [fresh tail]`; the
//! session-lifecycle hooks bind the conversation frontier.

use crate::compaction::run_compaction;
use crate::config::LcmConfig;
use crate::error::Result;
use crate::escalation::SummaryCircuitBreaker;
use crate::store::Store;
use crate::tokens::Tokenizer;
use async_trait::async_trait;
use daemon_common::SessionId;
use daemon_core::{Conversation, ContextEngine, ModelInfo, Pressure, Provider};
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
        // Snapshot the bits compaction needs, then run it without holding the state lock across the
        // aux-provider `await`s. The breaker is taken out and restored afterwards.
        let (tokenizer, session_id, mut breaker, first_compaction) = {
            let mut state = self.state.lock().expect("lcm state poisoned");
            let breaker = std::mem::take(&mut state.breaker);
            (
                state.tokenizer.clone(),
                state.session_id.clone(),
                breaker,
                state.compaction_count == 0,
            )
        };
        let session = if session_id.is_empty() {
            "unknown".to_string()
        } else {
            session_id
        };

        let (compacted, did_compact) = run_compaction(
            &self.store,
            &tokenizer,
            &self.config,
            self.aux.as_ref(),
            &mut breaker,
            &session,
            first_compaction,
            conv,
            Self::now(),
        )
        .await;

        let mut state = self.state.lock().expect("lcm state poisoned");
        state.breaker = breaker;
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
