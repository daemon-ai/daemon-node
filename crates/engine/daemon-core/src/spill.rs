// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Recoverable tool-result spilling (A1) — the §12 budget stage's lossless upgrade.
//!
//! `budget_result` truncation destroys bytes upstream of the conversation and every context
//! engine. This port lets the session's context engine offer a **spill store**
//! ([`ToolResultSpillStore`], exposed via [`ContextEngine::tool_spill`](crate::context::ContextEngine::tool_spill)):
//! an over-budget tool result is stored whole and replaced in the conversation by a head/tail
//! preview around a recovery notice, so nothing is lost and the model can pull the full bytes
//! back on demand. No store (the default engines), a store failure, or a notice that cannot fit
//! the cap all fall back to today's bounded truncation — spilling never makes a result larger,
//! never converts a successful call into an error, and never leaves a result unbounded.
//!
//! # Model Experience
//!
//! Within budget: unchanged. Over budget with a store: the model sees the first and last portions
//! of the (already fenced, if untrusted) output around one line —
//! `[Externalized tool output: <tool>, <dropped> bytes omitted; ref=<id>] <retrieval_hint>` —
//! where the hint tells it to call `lcm_expand` with the ref. Over budget without a store: the
//! existing `... [truncated N bytes over result budget]` marker.
//!
//! # KV Cache Effect
//!
//! The replacement is computed once, when the result is recorded; it is byte-stable thereafter,
//! so the cached prefix over prior turns is unaffected. Spilling shortens what enters the
//! conversation (cap-bounded), exactly like truncation did.
//!
//! # Durability / Replay Effect
//!
//! The durable conversation carries the replacement text, never the oversized original; the full
//! bytes live in the store (LCM's session-scoped externalization side-channel), keyed by the
//! `ref` embedded in the notice. LCM's replay/ingest machinery already recognizes the placeholder
//! grammar (`externalized_ref_regex`), so stored rows dedup instead of re-ingesting the preview.
//!
//! # Security Boundary
//!
//! Spilling runs AFTER the §12 untrusted-output fence, so the stored payload is the fenced form —
//! recovery via `lcm_expand` re-serves data the model was already allowed to see, wrapped exactly
//! as it first saw it. Refs are content-addressed bare file names (path-shaped refs are rejected
//! by the store) and recovery is session-scoped: another session's ref does not resolve.
//!
//! # Known Limitations and Deferred Work
//!
//! - Exempt tools ([`Tool::spill_exempt_for`](crate::tools::Tool::spill_exempt_for) or
//!   [`ToolResultSpillStore::exempt_tool`]) are hard-bounded by truncation, not unlimited.
//! - The head/tail split is byte-budgeted (char-boundary safe), not token-aware.
//! - The store is fire-and-forget per result: nothing garbage-collects a spill whose notice was
//!   later compacted away (LCM's transcript GC owns stored-payload lifecycle).

use crate::conversation::ToolResult;
use daemon_common::SessionId;

/// A stored spill's recovery handle: the content-addressed `ref` plus the model-facing hint that
/// names the recovery tool.
#[derive(Clone, Debug)]
pub struct SpillRef {
    /// The recovery reference (a bare file name in LCM's externalization store). Must contain no
    /// whitespace, `;`, or `]` — it is embedded in the bracketed notice the recovery regex parses.
    pub ref_id: String,
    /// One model-facing sentence naming how to recover the full bytes (e.g. call `lcm_expand`
    /// with the ref). Appended after the bracketed notice.
    pub retrieval_hint: String,
}

/// A spill-store failure (the pipeline logs it and falls back to truncation; it never fails the
/// tool call).
#[derive(Debug)]
pub struct SpillError(pub String);

impl std::fmt::Display for SpillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpillError {}

/// The tool-result spill port (A1): store an over-budget tool result whole so the §12 budget
/// stage can leave a recoverable notice instead of destroying bytes. Implemented by the LCM
/// context engine over its session-scoped externalization store; exposed to the pipeline through
/// [`ContextEngine::tool_spill`](crate::context::ContextEngine::tool_spill) and threaded on
/// [`TurnCx::spill`](crate::turn::TurnCx::spill).
#[async_trait::async_trait]
pub trait ToolResultSpillStore: Send + Sync {
    /// Persist `content` (the full, already-fenced result of `tool`'s call `call_id` in
    /// `session`) and return its recovery handle. An `Err` means the caller must fall back to
    /// truncation.
    async fn store(
        &self,
        session: &SessionId,
        call_id: &str,
        tool: &str,
        content: &str,
    ) -> Result<SpillRef, SpillError>;

    /// Whether results from tool `name` must never be spilled into this store (they stay
    /// hard-bounded by truncation). The store-side twin of
    /// [`Tool::spill_exempt_for`](crate::tools::Tool::spill_exempt_for), for exemptions only the
    /// store can know: LCM exempts its own `lcm_*` recovery tools so an oversized `lcm_expand`
    /// result never respills its own recovery (its pagination/`expand_hint` machinery is the
    /// retry path). Lives here rather than on the tool because the `lcm_*` tools reach the
    /// registry through host-side adapters this crate never sees.
    fn exempt_tool(&self, _name: &str) -> bool {
        false
    }
}

/// The spill notice (model-facing format fixed by the A1 plan; hygiene rule 6). The bracketed part
/// matches LCM's `externalized_ref_regex`, so its ingest/replay machinery recognizes the ref and
/// `lcm_expand` recovers it with zero new tools.
fn notice(tool: &str, dropped: usize, r: &SpillRef) -> String {
    format!(
        "[Externalized tool output: {tool}, {dropped} bytes omitted; ref={}] {}",
        r.ref_id, r.retrieval_hint
    )
}

/// Build the spilled replacement for `content` under `budget` bytes: a head/tail preview around
/// the recovery [`notice`]. The notice is priced worst-case (`dropped` can never exceed the full
/// length) and reserved inside the budget FIRST, so the replacement provably never exceeds
/// `budget`. Head gets `ceil(remainder/2)` bytes, tail `floor` (both snapped down to char
/// boundaries). `None` when the notice alone cannot fit — the caller falls back to truncation.
pub(crate) fn spill_replacement(
    content: &str,
    budget: usize,
    tool: &str,
    r: &SpillRef,
) -> Option<String> {
    // Two newline separators join head / notice / tail.
    let overhead = notice(tool, content.len(), r).len() + 2;
    if overhead > budget {
        return None;
    }
    let remainder = budget - overhead;
    let mut head_len = remainder.div_ceil(2).min(content.len());
    while head_len > 0 && !content.is_char_boundary(head_len) {
        head_len -= 1;
    }
    let tail_budget = remainder / 2;
    let mut tail_start = content.len().saturating_sub(tail_budget).max(head_len);
    while tail_start < content.len() && !content.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let head = &content[..head_len];
    let tail = &content[tail_start..];
    let dropped = content.len() - head.len() - tail.len();
    // The real notice (actual `dropped`) is never longer than the worst-case one priced above.
    Some(format!("{head}\n{}\n{tail}", notice(tool, dropped, r)))
}

/// The §12 sanitize+budget stage, spill-aware (A1): within `budget` (or `budget == 0`, cap
/// disabled) the result is untouched. Over budget, and unless `exempt`, the full content is
/// offered to `store`; on success the content becomes a head/tail preview around a recovery
/// notice. Store absent, store failure, exemption, or a notice that cannot fit all fall back to
/// the bounded truncation marker. Returns what happened plus the bytes no longer inline (for
/// tracing).
pub(crate) async fn budget_or_spill(
    result: &mut ToolResult,
    tool: &str,
    budget: usize,
    exempt: bool,
    store: Option<&dyn ToolResultSpillStore>,
    session: &SessionId,
) -> SpillDisposition {
    if budget == 0 || result.content.len() <= budget {
        return SpillDisposition::Unchanged;
    }
    if !exempt {
        if let Some(store) = store {
            match store
                .store(session, &result.call_id, tool, &result.content)
                .await
            {
                Ok(r) => {
                    if let Some(replacement) = spill_replacement(&result.content, budget, tool, &r)
                    {
                        let dropped = result.content.len() - replacement.len();
                        result.content = replacement;
                        return SpillDisposition::Spilled {
                            dropped,
                            ref_id: r.ref_id,
                        };
                    }
                    // Notice alone exceeds the cap: the payload is stored (harmless) but the
                    // inline form must stay bounded — truncate.
                }
                Err(e) => {
                    tracing::warn!(tool, error = %e, "engine.tool.spill_failed");
                }
            }
        }
    }
    match truncate_to_budget(result, budget) {
        Some(dropped) => SpillDisposition::Truncated { dropped },
        None => SpillDisposition::Unchanged,
    }
}

/// What [`budget_or_spill`] did to a result (tracing detail only).
pub(crate) enum SpillDisposition {
    /// Within budget (or cap disabled): untouched.
    Unchanged,
    /// Stored whole; the inline form is a head/tail preview + recovery notice.
    Spilled {
        /// Inline bytes given up relative to the original content.
        dropped: usize,
        /// The recovery ref embedded in the notice.
        ref_id: String,
    },
    /// Bounded destructive truncation (no store / store failed / exempt / notice could not fit).
    Truncated {
        /// Bytes destroyed over the budget.
        dropped: usize,
    },
}

/// The pre-A1 destructive truncation: cut to `budget` (char-boundary safe) and append the marker.
/// `0` disables the cap. The universal fallback — every over-budget result that does not spill
/// lands here, so no result is ever unbounded.
pub(crate) fn truncate_to_budget(result: &mut ToolResult, budget: usize) -> Option<usize> {
    if budget == 0 || result.content.len() <= budget {
        return None;
    }
    let mut cut = budget.min(result.content.len());
    while cut > 0 && !result.content.is_char_boundary(cut) {
        cut -= 1;
    }
    let dropped = result.content.len() - cut;
    result.content.truncate(cut);
    result.content.push_str(&format!(
        "\n... [truncated {dropped} bytes over result budget]"
    ));
    Some(dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spill_ref() -> SpillRef {
        SpillRef {
            ref_id: "tool_result_abc123_999_deadbeef.json".to_string(),
            retrieval_hint: "Full output stored; call lcm_expand with this ref to recover it."
                .to_string(),
        }
    }

    #[test]
    fn replacement_never_exceeds_budget() {
        let r = spill_ref();
        let content = "x".repeat(10_000);
        for budget in [200usize, 300, 500, 4096] {
            let replacement = spill_replacement(&content, budget, "shell", &r).unwrap();
            assert!(
                replacement.len() <= budget,
                "budget {budget}: replacement {} bytes",
                replacement.len()
            );
            assert!(replacement.contains("[Externalized tool output: shell,"));
            assert!(replacement.contains("ref=tool_result_abc123_999_deadbeef.json]"));
        }
    }

    /// A marginally-oversized result still holds the cap (the worst-case notice pricing cannot be
    /// cheated by a small overflow).
    #[test]
    fn marginally_oversized_holds_cap() {
        let r = spill_ref();
        let budget = 400usize;
        let content = "y".repeat(budget + 1);
        let replacement = spill_replacement(&content, budget, "fs", &r).unwrap();
        assert!(replacement.len() <= budget);
        // Head and tail both come from the original content.
        assert!(replacement.starts_with('y'));
        assert!(replacement.ends_with('y'));
    }

    #[test]
    fn notice_that_cannot_fit_returns_none() {
        let r = spill_ref();
        let content = "z".repeat(1000);
        assert!(spill_replacement(&content, 40, "shell", &r).is_none());
    }

    #[test]
    fn replacement_respects_char_boundaries() {
        let r = spill_ref();
        // 2-byte chars: budgets land mid-codepoint without boundary snapping.
        let content = "é".repeat(5000);
        for budget in [201usize, 250, 333] {
            let replacement = spill_replacement(&content, budget, "web", &r).unwrap();
            assert!(replacement.len() <= budget);
            assert!(replacement.contains("bytes omitted"));
        }
    }

    #[test]
    fn dropped_count_is_exact() {
        let r = spill_ref();
        let content: String = (0..2000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let replacement = spill_replacement(&content, 500, "shell", &r).unwrap();
        // head + tail bytes kept, notice reports exactly the difference.
        let kept: usize = replacement
            .split('\n')
            .filter(|line| !line.starts_with("[Externalized"))
            .map(|line| line.len())
            .sum();
        let reported: usize = replacement
            .split(", ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|n| n.parse().ok())
            .unwrap();
        assert_eq!(kept + reported, content.len());
    }

    #[test]
    fn truncate_matches_legacy_marker() {
        let mut result = ToolResult {
            call_id: "c1".into(),
            ok: true,
            content: "x".repeat(1000),
        };
        let dropped = truncate_to_budget(&mut result, 100).unwrap();
        assert_eq!(dropped, 900);
        assert!(result.content.starts_with(&"x".repeat(100)));
        assert!(result
            .content
            .contains("truncated 900 bytes over result budget"));
        // Within budget / disabled cap: untouched.
        let mut small = ToolResult {
            call_id: "c2".into(),
            ok: true,
            content: "ok".into(),
        };
        assert!(truncate_to_budget(&mut small, 100).is_none());
        assert!(truncate_to_budget(&mut small, 0).is_none());
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        // A multi-byte char straddling the cut point must not split mid-codepoint (no panic).
        let mut result = ToolResult {
            call_id: "c1".into(),
            ok: true,
            content: "é".repeat(100),
        };
        truncate_to_budget(&mut result, 51);
        assert!(result.content.contains("truncated"));
    }
}
