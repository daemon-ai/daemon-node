//! [`LcmContextEngine`] — the `daemon-core` [`ContextEngine`] implementation (skeleton).
//!
//! Maps LCM onto the §10 seam: `on_model` sizes the compaction threshold from the model window,
//! `before_turn` reports budget [`Pressure`], `compact` shrinks the body (recording a summary node),
//! and the session-lifecycle hooks warm/flush per incarnation. Compaction currently delegates to the
//! in-core drop-oldest strategy; the summary-DAG escalation grows from here.

use crate::compaction;
use crate::config::LcmConfig;
use crate::error::Result;
use crate::store::Store;
use async_trait::async_trait;
use daemon_common::SessionId;
use daemon_core::{estimate_tokens, ContextEngine, Conversation, ModelInfo, Pressure};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Mutable runtime state (small; behind a mutex so the sync hooks can update it).
#[derive(Default)]
struct State {
    /// The token threshold above which compaction triggers (derived from the model window).
    threshold_tokens: Option<usize>,
    /// The active session id (captured at `on_session_start`, used to key recorded summaries).
    session_id: String,
    /// How many compactions have run this incarnation.
    compaction_count: u64,
}

/// The LCM context engine over a single summary-store bank.
pub struct LcmContextEngine {
    config: LcmConfig,
    store: Store,
    state: Mutex<State>,
}

impl LcmContextEngine {
    /// Open the engine for the configured bank (in-memory when `config.db_path()` is `None`).
    pub fn open(config: LcmConfig) -> Result<Self> {
        let store = match config.db_path() {
            Some(path) => Store::open(path)?,
            None => Store::open_in_memory()?,
        };
        Ok(Self {
            config,
            store,
            state: Mutex::new(State::default()),
        })
    }

    /// Open an in-memory engine (tests / ephemeral nodes).
    pub fn open_in_memory() -> Result<Self> {
        Self::open(LcmConfig::in_memory())
    }

    /// The current compaction threshold (the model-derived budget), if known.
    fn threshold(&self) -> Option<usize> {
        self.state.lock().expect("lcm state poisoned").threshold_tokens
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
        if let Some(max) = model.max_context {
            let threshold = (max as f64 * self.config.threshold_percent) as usize;
            self.state.lock().expect("lcm state poisoned").threshold_tokens = Some(threshold);
        }
    }

    fn before_turn(&self, conv: &Conversation, budget: Option<usize>) -> Pressure {
        Pressure {
            used_tokens: estimate_tokens(conv),
            // Prefer the host's explicit budget; fall back to the model-derived threshold.
            budget_tokens: budget.or_else(|| self.threshold()),
        }
    }

    async fn compact(&self, conv: Conversation, budget: usize) -> Conversation {
        let before = estimate_tokens(&conv);
        let compacted = compaction::drop_oldest(conv, budget).await;
        let after = estimate_tokens(&compacted);
        if after < before {
            let (session_id, _count) = {
                let mut state = self.state.lock().expect("lcm state poisoned");
                state.compaction_count += 1;
                (state.session_id.clone(), state.compaction_count)
            };
            let session = if session_id.is_empty() {
                "unknown"
            } else {
                &session_id
            };
            // Skeleton: record the compaction as a depth-0 summary node. The deep port replaces the
            // placeholder text with a real escalated summary of the dropped span.
            if let Err(e) = self.store.record_summary(
                session,
                0,
                &format!("[compacted {} -> {} tokens]", before, after),
                0,
                (before.saturating_sub(after)) as i64,
                Self::now(),
            ) {
                tracing::warn!(error = %e, "lcm: failed to record summary node");
            }
        }
        compacted
    }

    fn on_session_start(&self, session: &SessionId) {
        self.state.lock().expect("lcm state poisoned").session_id = session.as_str().to_string();
    }

    fn on_session_end(&self, session: &SessionId, _conv: &Conversation) {
        let count = self.store.summary_count(session.as_str()).unwrap_or(0);
        tracing::debug!(session = %session, summaries = count, "lcm: session ended");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::{AssistantMsg, SystemPrompt};
    use daemon_protocol::UserMsg;

    fn convo(n: usize) -> Conversation {
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..n {
            c.push_user(UserMsg::new(format!("message number {i} ").repeat(20)));
            c.push_assistant(AssistantMsg::text(format!("reply number {i} ").repeat(20)));
        }
        c
    }

    #[tokio::test]
    async fn compaction_shrinks_and_records_a_summary() {
        let lcm = LcmContextEngine::open_in_memory().expect("open lcm");
        lcm.on_model(&ModelInfo {
            model: "test".into(),
            max_context: Some(1000),
        });
        lcm.on_session_start(&SessionId::new("s1"));
        let c = convo(10);
        let used = estimate_tokens(&c);
        let before = c.turns.len();
        let compacted = lcm.compact(c, used / 4).await;
        assert!(compacted.turns.len() < before, "older turns dropped");
        assert_eq!(lcm.store.summary_count("s1").unwrap(), 1);
    }
}
