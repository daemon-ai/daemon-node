// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The context engine seam (§10) — prompt composition, budget pressure, and compaction.
//!
//! The system prompt is a [`ComposedPrompt`]: typed ordered slots ([`SlotKind`]) rendered **once
//! per session** (session start/resume, or a model switch at a turn boundary) and reused
//! byte-for-byte every turn so provider prompt caching holds. Per-turn context (memory recall,
//! nudges) travels as a [`TurnInjection`] appended ephemerally to the outgoing request's last user
//! message — never the system prompt, never the durable conversation.
//!
//! Every turn's context is also measured against a token budget and (when over) compacted before
//! the model call. That is a swappable seam: the engine holds an `Arc<dyn ContextEngine>`
//! (defaulting to the cheap [`BudgetedContextEngine`]) and calls its hooks around the turn —
//! `before_turn` to measure [`Pressure`], `compact` to shrink the conversation, and
//! `after_response` to observe usage.
//!
//! This phase ships the seam + a real-but-cheap default ([`ContextStrategy::DropOldest`], which is
//! pair-preserving by construction since a [`Turn`] is atomic, with an anti-thrash guard). The
//! deep summarization strategy ([`ContextStrategy::Summarize`]) is provided but not default; the LCM
//! summary-DAG depth is a later phase.

use crate::conversation::{AssistantMsg, Conversation, Turn};
use crate::provider::{build_context, Provider, Request};
use async_trait::async_trait;
use daemon_common::{SessionId, UsageDelta};
use serde::{Deserialize, Serialize};
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

/// The typed slots a [`ComposedPrompt`] is composed from, in their fixed render order (§10).
///
/// The order is the composition contract: [`SlotKind::ORDER`] is the source of truth, and
/// [`ComposedPrompt::render`] emits non-empty slots exactly in that order. Slot *content* is
/// contributed by the engine's sources — the persona seeds `Identity`, [`StablePromptSource`]s and
/// [`ContextEngine::guidance_block`] fill `Guidance`/`ContextFiles`/`SkillsIndex`/`UserProfile`,
/// and memory `prompt_block()`s fill `MemoryBlock` — all captured once at composition time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SlotKind {
    /// The persona / identity text (SOUL.md once the persona store lands; the profile's system
    /// prompt today).
    Identity,
    /// Behavioral guidance blocks (core guidance, tool-use enforcement, context-engine notes).
    Guidance,
    /// Workspace context files (DAEMON.md / AGENTS.md / CLAUDE.md / .cursorrules).
    ContextFiles,
    /// The skills progressive-disclosure index.
    SkillsIndex,
    /// A memory provider's persistent prompt block.
    MemoryBlock,
    /// The per-profile user-profile (USER.md) snapshot.
    UserProfile,
    /// The date-only stamp.
    Stamp,
}

impl SlotKind {
    /// The canonical composition order (index = render order).
    pub const ORDER: [SlotKind; 7] = [
        SlotKind::Identity,
        SlotKind::Guidance,
        SlotKind::ContextFiles,
        SlotKind::SkillsIndex,
        SlotKind::MemoryBlock,
        SlotKind::UserProfile,
        SlotKind::Stamp,
    ];

    /// This kind's position in [`SlotKind::ORDER`].
    fn order_index(self) -> usize {
        SlotKind::ORDER
            .iter()
            .position(|k| *k == self)
            .unwrap_or(SlotKind::ORDER.len())
    }
}

/// One rendered slot of a [`ComposedPrompt`]: its kind plus the (non-empty) composed text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    /// The slot's kind (fixes its render position).
    pub kind: SlotKind,
    /// The slot's rendered text.
    pub text: String,
}

/// The session-scoped system-prompt composition (§10): typed ordered slots, rendered **once** per
/// session and reused byte-for-byte every turn so provider prompt caching holds.
///
/// Recomposition happens only at defined boundaries — session start, session resume (where the
/// stored composition is restored byte-identical), and a model switch at a turn boundary. Source
/// edits (skills, persona, memory snapshot) take effect at the *next* session, never mid-session.
/// Serializable because the engine persists the composition on the durable
/// [`Snapshot`](crate::Snapshot) to honor the byte-identical-restore invariant; it never travels
/// on the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposedPrompt {
    /// The composed slots: at most one per [`SlotKind`], held in [`SlotKind::ORDER`].
    slots: Vec<Slot>,
}

impl ComposedPrompt {
    /// Start building a composition ([`ComposedPromptBuilder`]).
    pub fn builder() -> ComposedPromptBuilder {
        ComposedPromptBuilder::default()
    }

    /// Render the composed system prompt: non-empty slot texts joined by `"\n\n"`, in
    /// [`SlotKind::ORDER`]. Byte-stable: the same composition always renders the same string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for slot in &self.slots {
            if slot.text.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&slot.text);
        }
        out
    }

    /// Per-slot byte attribution — for tests and tracing only, never on the wire. Every non-empty
    /// slot is reported; `sum(bytes)` plus the inter-slot separators equals `render().len()`.
    pub fn report(&self) -> Vec<SlotReport> {
        self.slots
            .iter()
            .filter(|s| !s.text.is_empty())
            .map(|s| SlotReport {
                kind: s.kind,
                bytes: s.text.len(),
            })
            .collect()
    }

    /// Whether the composition has no content at all.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.text.is_empty())
    }
}

/// One row of [`ComposedPrompt::report`]: a slot kind and how many bytes it contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotReport {
    /// The slot the bytes belong to.
    pub kind: SlotKind,
    /// The slot text's byte length.
    pub bytes: usize,
}

/// Accumulates per-slot contributions and builds a [`ComposedPrompt`] with the fixed slot order.
/// Multiple contributions to the same [`SlotKind`] are joined by `"\n\n"` in push order.
#[derive(Clone, Debug, Default)]
pub struct ComposedPromptBuilder {
    contributions: Vec<(SlotKind, String)>,
}

impl ComposedPromptBuilder {
    /// Contribute `text` to `kind`'s slot (empty text is ignored).
    pub fn push(&mut self, kind: SlotKind, text: impl Into<String>) -> &mut Self {
        let text = text.into();
        if !text.is_empty() {
            self.contributions.push((kind, text));
        }
        self
    }

    /// Build the composition: group contributions per kind (push order within a kind), join each
    /// group with `"\n\n"`, and order the resulting slots by [`SlotKind::ORDER`].
    pub fn build(self) -> ComposedPrompt {
        let mut slots: Vec<Slot> = Vec::new();
        for kind in SlotKind::ORDER {
            let text = self
                .contributions
                .iter()
                .filter(|(k, _)| *k == kind)
                .map(|(_, t)| t.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            if !text.is_empty() {
                slots.push(Slot { kind, text });
            }
        }
        debug_assert!(slots
            .windows(2)
            .all(|w| w[0].kind.order_index() < w[1].kind.order_index()));
        ComposedPrompt { slots }
    }
}

/// Per-turn ephemeral context (§10/§11): memory `recalled` blocks and engine `nudges`, appended to
/// the **last user message of the outgoing provider [`Request`] only** — never persisted to the
/// [`Conversation`](crate::Conversation) and never folded into the system prompt.
///
/// Cache tradeoff (deliberate): because the injection is rebuilt per turn, the previous turn's
/// user message loses its injected suffix on the next request, so the byte-stable cached prefix
/// ends just before the most recent previously-injected user message. The system prompt and all
/// deeper history stay cached; when the injection is empty the entire history is byte-stable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnInjection {
    /// Memory recalled for this specific turn (per-provider blocks, in provider order).
    pub recalled: Vec<String>,
    /// Engine nudges for this turn (e.g. the user-profile save nudge; empty until a source lands).
    pub nudges: Vec<String>,
}

impl TurnInjection {
    /// Whether there is nothing to inject this turn.
    pub fn is_empty(&self) -> bool {
        self.recalled.iter().all(|s| s.is_empty()) && self.nudges.iter().all(|s| s.is_empty())
    }

    /// Render the injection: recalled blocks then nudges, non-empty entries joined by `"\n\n"`.
    pub fn render(&self) -> String {
        self.recalled
            .iter()
            .chain(self.nudges.iter())
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Append the rendered injection to the **last `user` message** of `req` (the outgoing
    /// provider request only). No-op when the injection is empty or the request has no user
    /// message. The durable conversation is never touched.
    pub fn apply_to_last_user(&self, req: &mut Request) {
        if self.is_empty() {
            return;
        }
        if let Some(last_user) = req.messages.iter_mut().rev().find(|m| m.role == "user") {
            if !last_user.content.is_empty() {
                last_user.content.push_str("\n\n");
            }
            last_user.content.push_str(&self.render());
        }
    }
}

/// A source of a persistent system-prompt block (§10), independent of memory (§11).
///
/// The generic seam for any subsystem that wants an always-on block in the composed system prompt
/// without being a [`MemoryProvider`](crate::memory::MemoryProvider) — e.g. the skills *index*.
/// The engine captures each source's block **once at composition time** (session start/resume,
/// model switch), so the block **must be cache-stable** across a conversation: an edit takes
/// effect at the next session, and volatile content belongs in tool results, not here.
pub trait StablePromptSource: Send + Sync {
    /// The block to compose (`None` = nothing). Cheap: called once per composition.
    fn block(&self) -> Option<String>;

    /// The [`ComposedPrompt`] slot this source contributes to. Defaults to [`SlotKind::Guidance`];
    /// a source that owns a dedicated slot (the skills index, context files, the user profile)
    /// overrides this.
    fn slot_kind(&self) -> SlotKind {
        SlotKind::Guidance
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

    /// A static, cache-stable guidance contribution folded into the [`ComposedPrompt`]'s
    /// [`SlotKind::Guidance`] slot at composition time (session start / model switch). Default
    /// `None`. Must never change mid-session — a mid-session change would bust the cached prefix,
    /// which is exactly what this hook exists to avoid (e.g. LCM's tooling note is composed here
    /// from session start instead of being appended to the system prompt on first compaction).
    fn guidance_block(&self) -> Option<String> {
        None
    }

    /// Measure budget pressure for the conversation as it would be sent this turn.
    ///
    /// The conversation is `&mut` so a stateful engine can sanitize the provider-facing view in
    /// place before measuring (e.g. LCM's active-replay protection: secret redaction + runaway
    /// assistant-output quarantine). Implementations may rewrite turn *content* but must preserve
    /// the turn structure (count/order/pairing) — structural changes belong in
    /// [`ContextEngine::compact`]. The default engines leave the conversation untouched.
    fn before_turn(&self, conv: &mut Conversation, budget: Option<usize>) -> Pressure;

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

    /// A `/new`-style in-place reset of `session`'s context state (default no-op). The engine fires
    /// it when the conversation is cleared back to its root (a full-clear
    /// [`rewind_to`](crate::Engine::rewind_to)), so a stateful engine can reset its per-session
    /// state in step (LCM: retained-DAG prune by `new_session_retain_depth`, counters, ingest
    /// cursor) without the host having to remember an engine-specific inherent call.
    fn on_session_reset(&self, _session: &SessionId) {}

    /// An old → new session rollover (default no-op): finalize `old` (flushing `old_conv` when
    /// given), bind `new`, and — when `carry_over` — carry retained context across the boundary.
    /// The seam for LCM's ordered end→reset→start→carry-over (`rollover_session`,
    /// `LCM:engine.py:2240-2305`). No host trigger exists yet (a session-creation intent carrying
    /// the previous session id is a later workstream); the hook lands so engines stop exposing
    /// this as a host-must-remember inherent method.
    fn rollover_session(
        &self,
        _old: &SessionId,
        _new: &SessionId,
        _old_conv: Option<&Conversation>,
        _carry_over: bool,
    ) {
    }

    /// The advisory names of the context-management tools this engine owns (registered separately
    /// through the §12 [`ToolRegistry`](crate::tools)); empty by default.
    fn tools(&self) -> Vec<String> {
        Vec::new()
    }

    /// The [`CommandProvider`](crate::command::CommandProvider) view of this engine, when it also
    /// contributes operator/user commands (e.g. LCM's `/lcm`). Default `None`. Mirrors how the
    /// engine exposes tools through the §12 registry — a distinct seam from the model-facing
    /// [`ToolRegistry`](crate::tools), surfaced here so the node command registry can fold it in.
    /// A concrete engine that also `impl`s [`CommandProvider`](crate::command::CommandProvider)
    /// overrides this to `Some(self)`.
    fn command_provider(self: Arc<Self>) -> Option<crate::command::CommandProviderHandle> {
        None
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
    fn before_turn(&self, conv: &mut Conversation, budget: Option<usize>) -> Pressure {
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
mod composer_tests {
    use super::*;
    use crate::provider::RequestMsg;

    #[test]
    fn render_joins_slots_in_fixed_order() {
        let mut b = ComposedPrompt::builder();
        // Pushed deliberately out of order: build() must emit SlotKind::ORDER.
        b.push(SlotKind::Stamp, "STAMP");
        b.push(SlotKind::Identity, "IDENTITY");
        b.push(SlotKind::MemoryBlock, "MEMORY");
        b.push(SlotKind::Guidance, "GUIDANCE");
        let composed = b.build();
        assert_eq!(composed.render(), "IDENTITY\n\nGUIDANCE\n\nMEMORY\n\nSTAMP");
    }

    #[test]
    fn render_is_byte_stable() {
        let mut b = ComposedPrompt::builder();
        b.push(SlotKind::Identity, "persona ☤ with unicode");
        b.push(SlotKind::Guidance, "guide");
        let composed = b.build();
        let first = composed.render();
        assert_eq!(first.as_bytes(), composed.render().as_bytes());
        // A clone (the snapshot round-trip shape) renders the identical bytes.
        assert_eq!(composed.clone().render().as_bytes(), first.as_bytes());
    }

    #[test]
    fn multiple_contributions_to_one_slot_join_in_push_order() {
        let mut b = ComposedPrompt::builder();
        b.push(SlotKind::Guidance, "first");
        b.push(SlotKind::Guidance, "second");
        let composed = b.build();
        assert_eq!(composed.render(), "first\n\nsecond");
    }

    #[test]
    fn empty_contributions_are_ignored() {
        let mut b = ComposedPrompt::builder();
        b.push(SlotKind::Identity, "");
        b.push(SlotKind::Guidance, "g");
        let composed = b.build();
        assert_eq!(composed.render(), "g");
        assert_eq!(composed.report().len(), 1);
        assert!(!composed.is_empty());
        assert!(ComposedPrompt::default().is_empty());
        assert_eq!(ComposedPrompt::default().render(), "");
    }

    /// `report()` attributes every non-empty slot; the sum of slot bytes plus the `"\n\n"`
    /// separators equals the rendered length. The report is a plain method result — it never
    /// appears on a Request/Conversation (wire-invisible by construction).
    #[test]
    fn report_attributes_every_slot_and_sums_to_render_len() {
        let mut b = ComposedPrompt::builder();
        b.push(SlotKind::Identity, "abc");
        b.push(SlotKind::SkillsIndex, "skills-index");
        b.push(SlotKind::Stamp, "2026-07-09");
        let composed = b.build();
        let report = composed.report();
        assert_eq!(
            report.iter().map(|r| r.kind).collect::<Vec<_>>(),
            vec![SlotKind::Identity, SlotKind::SkillsIndex, SlotKind::Stamp],
        );
        let slot_bytes: usize = report.iter().map(|r| r.bytes).sum();
        let separators = (report.len() - 1) * "\n\n".len();
        assert_eq!(slot_bytes + separators, composed.render().len());
    }

    #[test]
    fn composed_prompt_round_trips_through_cbor() {
        // The snapshot persistence path: a CBOR round-trip restores byte-identical rendering.
        let mut b = ComposedPrompt::builder();
        b.push(SlotKind::Identity, "persona — ☤ 🦊");
        b.push(SlotKind::UserProfile, "# User\nlikes rust");
        let composed = b.build();
        let mut bytes = Vec::new();
        ciborium::into_writer(&composed, &mut bytes).unwrap();
        let decoded: ComposedPrompt = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(decoded, composed);
        assert_eq!(decoded.render().as_bytes(), composed.render().as_bytes());
    }

    fn msg(role: &str, content: &str) -> RequestMsg {
        RequestMsg {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn injection_appends_to_last_user_message_only() {
        let inj = TurnInjection {
            recalled: vec!["recalled memory".into()],
            nudges: vec!["a nudge".into()],
        };
        let mut req = Request {
            messages: vec![
                msg("user", "first"),
                msg("assistant", "reply"),
                msg("user", "second"),
                msg("assistant", "trailing"),
            ],
            ..Default::default()
        };
        inj.apply_to_last_user(&mut req);
        assert_eq!(req.messages[0].content, "first", "earlier user untouched");
        assert_eq!(req.messages[1].content, "reply");
        assert_eq!(
            req.messages[2].content, "second\n\nrecalled memory\n\na nudge",
            "recalled then nudges, appended to the LAST user message"
        );
        assert_eq!(req.messages[3].content, "trailing", "assistant untouched");
    }

    #[test]
    fn empty_injection_is_a_noop() {
        let inj = TurnInjection::default();
        assert!(inj.is_empty());
        let mut req = Request {
            messages: vec![msg("user", "hello")],
            ..Default::default()
        };
        inj.apply_to_last_user(&mut req);
        assert_eq!(req.messages[0].content, "hello");
        // Whitespace-free empties too.
        let inj = TurnInjection {
            recalled: vec![String::new()],
            nudges: vec![],
        };
        assert!(inj.is_empty());
    }

    #[test]
    fn injection_without_a_user_message_is_a_noop() {
        let inj = TurnInjection {
            recalled: vec!["mem".into()],
            nudges: vec![],
        };
        let mut req = Request {
            messages: vec![msg("assistant", "only assistant")],
            ..Default::default()
        };
        inj.apply_to_last_user(&mut req);
        assert_eq!(req.messages[0].content, "only assistant");
    }

    #[test]
    fn stable_prompt_source_defaults_to_guidance_slot() {
        struct Fixed;
        impl StablePromptSource for Fixed {
            fn block(&self) -> Option<String> {
                Some("x".into())
            }
        }
        assert_eq!(Fixed.slot_kind(), SlotKind::Guidance);
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
        let mut c = convo(10);
        let used = estimate_tokens(&c);
        let eng = BudgetedContextEngine::default();
        assert!(eng.before_turn(&mut c, Some(used / 2)).over_budget());
        assert!(!eng.before_turn(&mut c, Some(used * 2)).over_budget());
        assert!(!eng.before_turn(&mut c, None).over_budget());
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
