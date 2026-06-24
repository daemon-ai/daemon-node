//! The memory provider seam (§11) — recall + persistent prompt blocks across turns/sessions.
//!
//! Memory is modeled as a *set* of providers the engine consults at fixed hook points in the turn
//! loop, so different memory backends (a flat MEMORY.md, the Mnemosyne BEAM engine, a future vector
//! store) compose without re-shaping the loop. The engine holds `Vec<Arc<dyn MemoryProvider>>`
//! (empty by default — memory is opt-in) and drives the §11 hook order:
//!
//! `recall -> prompt_block (into the stable tier) -> before_compact -> compact -> assemble ->
//!  after_turn -> after_response`.
//!
//! Design notes (the seam is deliberately narrow):
//! - A provider **owns its recall**: it ranks/formats internally and returns one [`RecalledBlock`]
//!   ready to inject, rather than handing the engine scored fragments to re-rank across providers.
//! - The persist/consolidate hooks are **async** because a real backend's writes are I/O.
//! - Memory does *not* expose model tools here. A backend that wants `remember`/`recall` tools
//!   registers them through the §12 [`ToolRegistry`](crate::tools) like any other tool; the memory
//!   seam stays about context, not dispatch.
//!
//! This phase ships the seam + a minimal builtin [`FileMemory`] (a frozen MEMORY.md snapshot served
//! as a stable prompt block, with naive paragraph recall) as a *provided, non-default* impl that
//! proves the seam end-to-end.

use crate::conversation::{Conversation, Turn};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

/// What the engine asks a memory provider to recall for a turn (§11).
#[derive(Clone, Debug, Default)]
pub struct RecallQuery {
    /// The salient query text for the turn (typically the latest user message).
    pub text: String,
    /// The maximum number of memories the provider should fold into its block.
    pub top_k: usize,
}

/// Why a session boundary was crossed (the `on_session_switch` reason, §11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchReason {
    /// A fresh session is starting.
    Start,
    /// The switch is a context compaction event.
    Compaction,
    /// Work is being handed off to/from another session (e.g. a turn suspending to a delegation).
    Handoff,
    /// A suspended session is resuming (a background completion re-activates the incarnation).
    Resume,
    /// The session is ending (the host is tearing the incarnation down).
    End,
    /// An operator-initiated switch.
    Manual,
}

/// A persistent block a provider injects into the **stable** prompt tier every turn (§11).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptBlock {
    /// The block text.
    pub text: String,
}

/// A block a provider recalls into the **recalled** prompt tier for one turn (§11) — already ranked
/// and formatted by the provider, ready to inject.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecalledBlock {
    /// The formatted recall text to inject into the prompt.
    pub text: String,
}

/// The §11 memory seam. All hooks have no-op defaults so a provider implements only what it needs.
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// A stable identifier (for diagnostics / dedup).
    fn name(&self) -> &str;

    /// A persistent block injected into the stable prompt tier every turn (`None` = no block).
    fn prompt_block(&self) -> Option<PromptBlock> {
        None
    }

    /// Recall a block relevant to `query` for this turn (`None` = nothing relevant). The provider
    /// owns ranking + formatting.
    async fn recall(&self, _query: &RecallQuery) -> Option<RecalledBlock> {
        None
    }

    /// Observe a completed turn (to persist new memories). Called after the turn is recorded.
    async fn after_turn(&self, _turn: &Turn, _conv: &Conversation) {}

    /// The context is about to be compacted (a chance to persist salient facts before loss).
    async fn before_compact(&self, _conv: &Conversation) {}

    /// The session boundary was crossed (a chance to consolidate). The provider already knows its
    /// own session/bank (it is constructed per-session by the composition layer), so the reason —
    /// not a session id — is all the engine threads here.
    async fn on_session_switch(&self, _reason: SwitchReason) {}

    /// The [`CommandProvider`](crate::command::CommandProvider) view of this provider, when it also
    /// contributes operator/user commands (e.g. Mnemosyne's `/memory`). Default `None`. A distinct
    /// seam from the model-facing tools a backend may register through the §12
    /// [`ToolRegistry`](crate::tools); surfaced here so the node command registry can fold it in. A
    /// concrete provider that also `impl`s [`CommandProvider`](crate::command::CommandProvider)
    /// overrides this to `Some(self)`.
    fn command_provider(self: Arc<Self>) -> Option<crate::command::CommandProviderHandle> {
        None
    }
}

/// A minimal builtin memory provider (§11): a frozen MEMORY.md snapshot served as a stable prompt
/// block, with naive paragraph-level recall. It is read-only (a frozen volatile tier) — proving the
/// seam without a write path. Provided, not default.
pub struct FileMemory {
    name: String,
    snapshot: String,
}

impl FileMemory {
    /// Load a frozen snapshot from `path` (an empty snapshot if the file is absent/unreadable).
    pub fn load(path: impl AsRef<Path>) -> Self {
        let snapshot = std::fs::read_to_string(path.as_ref()).unwrap_or_default();
        Self {
            name: "file-memory".into(),
            snapshot,
        }
    }

    /// Construct directly from an in-memory snapshot (test/embedding convenience).
    pub fn from_snapshot(snapshot: impl Into<String>) -> Self {
        Self {
            name: "file-memory".into(),
            snapshot: snapshot.into(),
        }
    }

    /// The non-empty paragraphs of the snapshot.
    fn paragraphs(&self) -> impl Iterator<Item = &str> {
        self.snapshot
            .split("\n\n")
            .map(str::trim)
            .filter(|p| !p.is_empty())
    }
}

#[async_trait]
impl MemoryProvider for FileMemory {
    fn name(&self) -> &str {
        &self.name
    }

    fn prompt_block(&self) -> Option<PromptBlock> {
        let trimmed = self.snapshot.trim();
        (!trimmed.is_empty()).then(|| PromptBlock {
            text: format!("# Memory\n{trimmed}"),
        })
    }

    async fn recall(&self, query: &RecallQuery) -> Option<RecalledBlock> {
        let terms: Vec<String> = query
            .text
            .to_ascii_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(str::to_string)
            .collect();
        if terms.is_empty() {
            return None;
        }
        let mut scored: Vec<(usize, &str)> = self
            .paragraphs()
            .filter_map(|p| {
                let lower = p.to_ascii_lowercase();
                let hits = terms.iter().filter(|t| lower.contains(*t)).count();
                (hits > 0).then_some((hits, p))
            })
            .collect();
        if scored.is_empty() {
            return None;
        }
        scored.sort_by_key(|(hits, _)| std::cmp::Reverse(*hits));
        scored.truncate(query.top_k.max(1));
        let text = scored
            .iter()
            .map(|(_, p)| *p)
            .collect::<Vec<_>>()
            .join("\n");
        Some(RecalledBlock { text })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn file_memory_serves_prompt_block_and_recalls() {
        let mem = FileMemory::from_snapshot(
            "The deploy key lives in vault.\n\nThe build uses nix develop.\n\nUnrelated note.",
        );
        let block = mem.prompt_block().expect("a prompt block");
        assert!(block.text.contains("deploy key"));

        let hit = mem
            .recall(&RecallQuery {
                text: "how do I deploy the key".into(),
                top_k: 5,
            })
            .await
            .expect("a recalled block");
        assert!(hit.text.contains("deploy key"));
    }

    #[tokio::test]
    async fn empty_snapshot_has_no_block() {
        let mem = FileMemory::from_snapshot("   ");
        assert!(mem.prompt_block().is_none());
        assert!(mem
            .recall(&RecallQuery {
                text: "anything".into(),
                top_k: 3
            })
            .await
            .is_none());
    }
}
