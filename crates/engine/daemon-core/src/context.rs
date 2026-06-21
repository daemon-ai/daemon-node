//! The context engine seam (§10) — prompt assembly, budget pressure, and compaction.
//!
//! Every turn's prompt is assembled, measured against a token budget, and (when over) compacted
//! before the model call. This module establishes that as a swappable seam: the engine holds an
//! `Arc<dyn ContextEngine>` (defaulting to the cheap [`BudgetedContextEngine`]) and calls its hooks
//! around the turn — `before_turn` to measure [`Pressure`], `compact` to shrink the conversation,
//! and `after_response` to observe usage. The [`PromptAssembler`] tiers (stable / blocks / recalled /
//! body) are where memory (§11) injects recalled context without re-shaping the turn loop.
//!
//! This phase ships the seam + a real-but-cheap default ([`ContextStrategy::DropOldest`], which is
//! pair-preserving by construction since a [`Turn`] is atomic, with an anti-thrash guard). The
//! deep summarization strategy ([`ContextStrategy::Summarize`]) is provided but not default; the LCM
//! summary-DAG depth is a later phase.

use crate::conversation::{AssistantMsg, Conversation, Turn};
use crate::provider::{build_context, Provider, Request};
use async_trait::async_trait;
use daemon_common::{SessionId, UsageDelta};
use std::sync::Arc;

/// A cheap token estimate: ~4 chars/token plus a small per-turn structural overhead. Good enough to
/// drive budget pressure without a tokenizer dependency (the budget is a soft guard, not billing).
pub fn estimate_tokens(conv: &Conversation) -> usize {
    let mut chars = conv.system.text.len();
    for turn in &conv.turns {
        chars += estimate_turn_chars(turn);
    }
    chars / 4
}

fn estimate_turn_chars(turn: &Turn) -> usize {
    // A small per-message overhead approximates role/framing tokens.
    const OVERHEAD: usize = 16;
    match turn {
        Turn::User(u) => u.text.len() + OVERHEAD,
        Turn::Assistant(a) => {
            a.text.len() + a.reasoning.as_ref().map(|r| r.len()).unwrap_or(0) + OVERHEAD
        }
        Turn::Tool(t) => {
            let mut n = t.assistant.text.len() + OVERHEAD;
            for (call, result) in &t.calls {
                n += call.name.len() + call.args.len() + result.content.len() + OVERHEAD;
            }
            n
        }
    }
}

/// The context-budget pressure measured before a turn (§10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pressure {
    /// The estimated tokens the assembled context would use.
    pub used_tokens: usize,
    /// The configured soft budget, if any.
    pub budget_tokens: Option<usize>,
}

impl Pressure {
    /// Whether the estimate exceeds the budget (so the turn should compact first).
    pub fn over_budget(&self) -> bool {
        matches!(self.budget_tokens, Some(b) if self.used_tokens > b)
    }
}

/// The tiered prompt assembler (§10): persistent `stable` blocks, memory `blocks`, per-turn
/// `recalled` context, and the conversation `body`. Memory (§11) populates the non-body tiers; the
/// assembler folds them into the request's system preamble ahead of the flattened body.
#[derive(Clone, Debug, Default)]
pub struct PromptAssembler {
    /// Persistent system-tier blocks (e.g. a memory provider's always-on `prompt_block`).
    pub stable: Vec<String>,
    /// Additional prompt blocks injected this turn.
    pub blocks: Vec<String>,
    /// Memory recalled for this specific turn.
    pub recalled: Vec<String>,
}

impl PromptAssembler {
    /// Assemble the [`Request`]: the flattened conversation body with the stable/blocks/recalled
    /// tiers folded into the system preamble (in tier order).
    pub fn assemble(&self, conv: &Conversation, tools: &[crate::tools::ToolDef]) -> Request {
        let mut req = build_context(conv, tools);
        let mut system = req.system;
        for extra in self
            .stable
            .iter()
            .chain(self.blocks.iter())
            .chain(self.recalled.iter())
            .filter(|s| !s.is_empty())
        {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(extra);
        }
        req.system = system;
        req
    }

    /// Drop the per-turn tiers (kept between turns: `stable` persists, `blocks`/`recalled` are
    /// re-gathered each turn by memory).
    pub fn reset_turn(&mut self) {
        self.blocks.clear();
        self.recalled.clear();
    }
}

/// A lightweight description of the active model, handed to a [`ContextEngine`] via
/// [`ContextEngine::on_model`] so a stateful engine can size its token budgets/thresholds from the
/// real context window (e.g. LCM's compaction threshold). Best-effort: `max_context` is `None` when
/// the provider does not declare a window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelInfo {
    /// The profile/model identifier the engine is running under (a best-effort label, e.g. the
    /// active profile name — providers do not yet surface a canonical model id).
    pub model: String,
    /// The model's maximum context window in tokens, if the provider declares one
    /// ([`crate::provider::Capabilities::max_context`]).
    pub max_context: Option<u32>,
}

/// The context-engine seam (§10). The engine calls these hooks around each turn; the default impl is
/// stateless so it is trivially `Send + Sync` and cheap to share.
///
/// Lifecycle ordering (driven by the engine): `on_model` + `on_session_start` fire once before the
/// first turn; `before_turn`/`compact`/`after_response` fire around each turn; `on_session_end` fires
/// when the host tears the session down ([`crate::Engine::end_session`]).
///
/// Tools are intentionally *not* dispatched through this seam: an engine that exposes drill-down
/// tools (e.g. LCM's `lcm_*`) registers them through the §12 [`ToolRegistry`](crate::tools) like any
/// other tool, holding an `Arc` to itself in the closure. [`ContextEngine::tools`] returns only the
/// advisory names it owns (for diagnostics / dedup), never a dispatch entrypoint.
#[async_trait]
pub trait ContextEngine: Send + Sync {
    /// Observe the active model so the engine can size budgets/thresholds (default no-op). Called
    /// once before the first turn (and again if the host swaps the model).
    fn on_model(&self, _model: &ModelInfo) {}

    /// Measure budget pressure for the conversation as it would be sent this turn.
    fn before_turn(&self, conv: &Conversation, budget: Option<usize>) -> Pressure;

    /// Compact `conv` to fit `budget` tokens, returning the (possibly unchanged) conversation. Must
    /// preserve tool-call/result pairing — operating on whole [`Turn`]s does this by construction.
    async fn compact(&self, conv: Conversation, budget: usize) -> Conversation;

    /// Observe the usage a model response accrued (for adaptive budgeting; default no-op).
    fn after_response(&self, _usage: &UsageDelta) {}

    /// Session-lifecycle hooks (default no-ops; a stateful engine warms/flushes here).
    /// `on_session_start` fires once before the first turn of an incarnation.
    fn on_session_start(&self, _session: &SessionId) {}
    /// See [`ContextEngine::on_session_start`]. Fires on session teardown with the final
    /// conversation so the engine can flush a closing summary.
    fn on_session_end(&self, _session: &SessionId, _conv: &Conversation) {}

    /// The advisory names of the context-management tools this engine owns (registered separately
    /// through the §12 [`ToolRegistry`](crate::tools)); empty by default.
    fn tools(&self) -> Vec<String> {
        Vec::new()
    }
}

/// How [`BudgetedContextEngine`] compacts an over-budget conversation.
#[derive(Clone)]
pub enum ContextStrategy {
    /// Drop the oldest turns (cheap, pair-preserving) with an anti-thrash guard. The default.
    DropOldest,
    /// Summarize the dropped prefix via an auxiliary provider, replacing it with one synthetic
    /// assistant turn. Provided but not the default (it costs an extra model call); falls back to
    /// drop-oldest if the auxiliary call fails.
    Summarize {
        /// The auxiliary provider used to produce the summary.
        aux: Arc<dyn Provider>,
    },
}

impl std::fmt::Debug for ContextStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextStrategy::DropOldest => f.write_str("DropOldest"),
            ContextStrategy::Summarize { .. } => f.write_str("Summarize"),
        }
    }
}

/// The default context engine: estimate-driven budget pressure + a configurable compaction strategy
/// (drop-oldest by default) with an anti-thrash guard that skips compaction when it would free less
/// than 10% of the current context.
#[derive(Clone, Debug)]
pub struct BudgetedContextEngine {
    strategy: ContextStrategy,
}

impl Default for BudgetedContextEngine {
    fn default() -> Self {
        Self {
            strategy: ContextStrategy::DropOldest,
        }
    }
}

impl BudgetedContextEngine {
    /// A context engine using `strategy`.
    pub fn new(strategy: ContextStrategy) -> Self {
        Self { strategy }
    }

    /// Plan how many of the oldest turns to drop to bring the estimate under `budget`, never
    /// dropping the most recent turn. Returns `(drop_count, used_after)`.
    fn plan_drop(conv: &Conversation, budget: usize) -> (usize, usize) {
        let mut used = estimate_tokens(conv);
        let mut drop = 0usize;
        let last_keepable = conv.turns.len().saturating_sub(1);
        while drop < last_keepable && used > budget {
            used = used.saturating_sub(estimate_turn_chars(&conv.turns[drop]) / 4);
            drop += 1;
        }
        (drop, used)
    }

    /// The shared drop-oldest core (also the fallback for [`ContextStrategy::Summarize`]).
    fn drop_oldest(conv: Conversation, budget: usize) -> Conversation {
        let used_before = estimate_tokens(&conv);
        let (drop, used_after) = Self::plan_drop(&conv, budget);
        // Anti-thrash: skip if compaction frees < 10% of the current context (not worth disrupting
        // the cache / paying a summary for a marginal gain).
        if drop == 0 || (used_before.saturating_sub(used_after)) * 10 < used_before {
            return conv;
        }
        let Conversation { system, turns } = conv;
        Conversation {
            system,
            turns: turns.into_iter().skip(drop).collect(),
        }
    }
}

#[async_trait]
impl ContextEngine for BudgetedContextEngine {
    fn before_turn(&self, conv: &Conversation, budget: Option<usize>) -> Pressure {
        Pressure {
            used_tokens: estimate_tokens(conv),
            budget_tokens: budget,
        }
    }

    async fn compact(&self, conv: Conversation, budget: usize) -> Conversation {
        match &self.strategy {
            ContextStrategy::DropOldest => Self::drop_oldest(conv, budget),
            ContextStrategy::Summarize { aux } => {
                let used_before = estimate_tokens(&conv);
                let (drop, used_after) = Self::plan_drop(&conv, budget);
                if drop == 0 || (used_before.saturating_sub(used_after)) * 10 < used_before {
                    return conv;
                }
                // Summarize the dropped prefix; on any failure fall back to a plain drop.
                let prefix = Conversation {
                    system: Default::default(),
                    turns: conv.turns[..drop].to_vec(),
                };
                let mut req = build_context(&prefix, &[]);
                req.system =
                    "Summarize the following conversation prefix into a compact note that \
                     preserves decisions, facts, and open threads. Be terse."
                        .into();
                match aux.chat(req).await {
                    Ok(out) if !out.text.is_empty() => {
                        let Conversation { system, turns } = conv;
                        let mut kept: Vec<Turn> = Vec::with_capacity(turns.len() - drop + 1);
                        kept.push(Turn::Assistant(AssistantMsg::text(format!(
                            "[summary of {drop} earlier turns]\n{}",
                            out.text
                        ))));
                        kept.extend(turns.into_iter().skip(drop));
                        Conversation {
                            system,
                            turns: kept,
                        }
                    }
                    _ => Self::drop_oldest(conv, budget),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::SystemPrompt;
    use daemon_protocol::UserMsg;

    fn convo(n: usize) -> Conversation {
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        for i in 0..n {
            c.push_user(UserMsg::new(format!("message number {i} ").repeat(20)));
            c.push_assistant(AssistantMsg::text(format!("reply number {i} ").repeat(20)));
        }
        c
    }

    #[test]
    fn pressure_flags_over_budget() {
        let c = convo(10);
        let used = estimate_tokens(&c);
        let eng = BudgetedContextEngine::default();
        assert!(eng.before_turn(&c, Some(used / 2)).over_budget());
        assert!(!eng.before_turn(&c, Some(used * 2)).over_budget());
        assert!(!eng.before_turn(&c, None).over_budget());
    }

    #[tokio::test]
    async fn drop_oldest_shrinks_and_preserves_recent() {
        let c = convo(10);
        let before = c.turns.len();
        let used = estimate_tokens(&c);
        let eng = BudgetedContextEngine::default();
        let compacted = eng.compact(c.clone(), used / 4).await;
        assert!(compacted.turns.len() < before, "older turns were dropped");
        // The most recent turn survives.
        assert_eq!(compacted.turns.last(), c.turns.last());
        assert_eq!(compacted.system, c.system);
        assert!(estimate_tokens(&compacted) <= used);
    }

    #[tokio::test]
    async fn anti_thrash_skips_marginal_compaction() {
        let c = convo(10);
        let used = estimate_tokens(&c);
        let eng = BudgetedContextEngine::default();
        // A budget just 1 token under current: dropping one small turn frees < 10% -> skip.
        let compacted = eng.compact(c.clone(), used.saturating_sub(1)).await;
        assert_eq!(
            compacted.turns.len(),
            c.turns.len(),
            "marginal compaction skipped"
        );
    }
}
