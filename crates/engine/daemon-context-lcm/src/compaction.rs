//! Compaction (skeleton).
//!
//! The skeleton delegates the actual shrink to the in-core, tested
//! [`BudgetedContextEngine`](daemon_core::context::BudgetedContextEngine) drop-oldest strategy
//! (pair-preserving, anti-thrash). The deep port replaces this with the summary-DAG escalation
//! (`daemon-context-lcm-port-spec.md` §6): summarize the dropped prefix into a [`SummaryNode`] and
//! splice it back as a synthetic turn.

use daemon_core::context::{BudgetedContextEngine, ContextEngine};
use daemon_core::Conversation;

/// Compact `conv` to fit `budget` tokens by dropping the oldest turns (delegating to the in-core
/// budgeted engine). Returns the possibly-unchanged conversation.
pub(crate) async fn drop_oldest(conv: Conversation, budget: usize) -> Conversation {
    BudgetedContextEngine::default().compact(conv, budget).await
}
