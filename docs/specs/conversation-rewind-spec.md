# conversation rewind — the NodeApi extension specification

This spec describes the **conversation-rewind primitive** the `daemon` NodeApi exposes so a client
(the desktop GUI / TUI in `../daemon-app`) can let a user *rewind the conversation to a prior user
message and redo from there* — restore (re-run the same text), edit (re-run edited text), or
regenerate the assistant reply — end-to-end against a live engine, instead of only client-side.

**Implementation status.** Implemented for live `daemon-core`-backed sessions. The client-facing
contract is `ControlApi::rewind(session, RewindPoint { anchor, restore_workspace })`
(`ApiRequest::Rewind`), introduced in wire v14 and carried by the current wire envelope. It funnels to
the same live-engine rewind path as the lower-level `AgentCommand::RewindTo`, with interrupt-first
semantics, in-place snapshot reconstruction, an epoch bump, an append-only durable journal seal, and
optional workspace-checkpoint rollback. Foreign ACP-backed sessions are **explicitly non-rewindable**
(ACP has no truncate-at-anchor primitive — see §9); they advertise `SessionInfo::rewindable = false`
so a client hides rewind for them rather than issuing a command that is dropped. Durable
non-resident sessions currently return `Unsupported` until the re-incarnation truncate-and-reseal path
lands.

It builds on:

- [`daemon-host-spec.md`](daemon-host-spec.md) — the durable activation layer, snapshot contract.
- [`daemon-lifecycle-persistence.md`](daemon-lifecycle-persistence.md) — snapshot / journal durability.
- [`daemon-orchestrator-spec.md`](daemon-orchestrator-spec.md) — §17 surface translation.
- [`daemon-gui-readiness-roadmap.md`](daemon-gui-readiness-roadmap.md) — the GUI-facing gap list (`no fork_session`, etc.).

## 1. Problem solved

The original gap was that the engine was append-only and had no conversation-rewind op:

- `Conversation { turns: Vec<Turn> }` (`crates/engine/daemon-core/src/conversation.rs`) only grows.
- The write surface is `AgentCommand::{ StartTurn, Steer, Observe, Interrupt, Snapshot, Shutdown }`
  (`crates/contracts/daemon-protocol/src/lib.rs`). `StartTurn` always **appends** a turn; `Snapshot`
  is a read-only `ConvView`; `Interrupt` ends the live turn but does not truncate history.
- HITL is **one-shot**: approvals/clarifies arrive as `Outbound::Request(HostRequestKind::{Approval,
  Input,Choice,…})` and are closed by `SessionApi::respond` / `ControlApi::approval_decide`
  (request row idempotent, removed once answered). There is **no reopen-by-id**.
- `ControlApi::checkpoints` + `checkpoint_rewind` (`crates/contracts/daemon-api/src/lib.rs`) restore
  the **filesystem** from pre-mutating-tool snapshots only — **not** conversation / journal state.
- Read cursors (`session_history(after_cursor)`, the live-log `segment`) are read-only; there is no
  truncate-to-cursor, edit, fork, or regenerate op.

That gap is closed for live daemon-core sessions by `ControlApi::rewind`, backed by the same engine
primitive described below. The remaining limitation is non-resident durable sessions: the host has a
sealed journal but no resident engine to truncate, so it returns `Unsupported("rewind of a
non-resident durable session...")` rather than silently no-oping.

## 2. Client op — `ControlApi::rewind`

Clients call the unified control-plane operation:

```rust
async fn rewind(&self, session: SessionId, point: RewindPoint) -> Result<(), ApiError>;

pub struct RewindPoint {
    pub anchor: RewindAnchor,
    pub restore_workspace: bool,
}
```

This replaces separate client flows for conversation rewind and workspace-only checkpoint rewind. For
live daemon-core sessions the host translates it into the engine-level command below; for non-resident
durable sessions it returns `Unsupported` until the activation-driven re-incarnation path exists.

### Engine command — `RewindTo`

The engine-level submit-path command remains the mutation primitive:

```rust
pub enum AgentCommand {
    // … existing variants …

    /// Seal/truncate the conversation after `anchor` and restore the engine snapshot
    /// for that point, so a subsequent StartTurn replays from there. Interrupt-first:
    /// if a turn is live the engine interrupts it before truncating.
    RewindTo {
        /// Where to rewind to (resolves to a durable journal cursor / turn ordinal).
        anchor: RewindAnchor,
        /// Correlation id (echoed on AgentEvent::Rewound).
        request_id: ReqId,
    },
}

/// A durable, replay-stable address of a rewind point.
pub enum RewindAnchor {
    /// Truncate so the turn that produced this user message is the new tail-to-redo:
    /// everything from that user turn onward (inclusive) is sealed off.
    UserTurn { ordinal: u64 },
    /// Truncate the assistant reply that followed `ordinal` but keep the user turn
    /// (the regenerate case): everything after the user turn is sealed off.
    ReplyAfter { ordinal: u64 },
    /// A raw durable journal cursor (the §17 `session_history` cursor), for clients
    /// that address by cursor rather than ordinal.
    Cursor { seq: u64 },
}
```

Semantics, in order:

1. **Interrupt-first.** If a turn is live, behave as `Interrupt { reason: "rewind" }` first.
2. **Seal/truncate.** Drop `turns` after the anchor (for `UserTurn`, drop the anchor turn too).
   This is a *journal truncation* — see §6 for the sealed-segment vs hard-delete choice.
3. **Restore snapshot.** Reload the engine `Snapshot` captured for the anchor's epoch/segment so
   context-window state, tool state, and usage counters match the rewound point (not the live tail).
4. **Bump epoch.** Increment the session `Epoch` so any in-flight commits/events from the
   interrupted turn that arrive late are fenced and dropped (§6).
5. **Resume.** The engine is now idle at the rewound point, ready for the client's next `StartTurn`.

## 3. New event — `Rewound`

```rust
pub enum AgentEvent {
    // … existing variants …

    /// The conversation was rewound; live clients drop their tail at/after `to_cursor`
    /// before the replayed TurnStarted arrives.
    Rewound {
        seq: u64,
        /// Echoed from RewindTo.
        request_id: ReqId,
        /// The retained conversation length in turns — the new tail ordinal.
        to_cursor: u64,
        /// The new epoch fencing stale commits/events.
        epoch: u64,
    },
}
```

`to_cursor` is the **retained turn ordinal** — the new `conversation.turns.len()` after truncation,
the same coordinate space as `RewindAnchor::UserTurn { ordinal }` and as `ConvView::turns` indices.
The engine addresses turns by ordinal (it does not assign journal cursors — those are host-side), so
a live client drops every turn it holds with ordinal `>= to_cursor` the moment `Rewound` arrives, and
the UI matches the engine before the replayed `TurnStarted { trigger: User }` streams in. A
reconnecting client reconciles against the engine's truncated conversation — the authoritative
`Snapshot` / `ConvView` — and consults `JournalPageView::sealed_after` for the durable audit boundary
(§6).

## 4. Action mapping

The three client actions decompose into `ControlApi::rewind` + a normal turn command:

| Client action | NodeApi sequence |
| :--- | :--- |
| **restore** (re-run same text) | `ControlApi::rewind(UserTurn(o))` → `StartTurn { input: <original UserMsg> }` |
| **edit** (re-run edited text) | `ControlApi::rewind(UserTurn(o))` → `StartTurn { input: <edited UserMsg> }` |
| **regenerate** (new reply, same prompt) | `ControlApi::rewind(ReplyAfter(o))` → `StartTurn` re-run **without** re-appending the user turn |

`restore`/`edit` differ only in the `UserMsg` payload of the follow-up `StartTurn`; `regenerate`
keeps the user turn (`ReplyAfter`) and re-runs without a new input.

## 5. Future extension (out of current scope): re-answering an interactive block

Re-opening an **already-answered** approval / clarify in place (reset it to awaiting, re-answer,
stream a fresh follow-up) is **not implementable** on this backend and is **out of scope** for the
client work this spec accompanies:

- HITL is one-shot — `respond` / `approval_decide` close the `HostRequest` row idempotently; there
  is no "unanswer" and no reopen-by-id.
- The conversation is append-only, so the answered turn is durable history, not mutable state.

The only sound model is to treat it as a rewind: `RewindTo` **the turn that raised the
`HostRequest`**, then re-run that turn so the engine re-emits the `Outbound::Request` fresh. That
needs the §2 primitive plus a way to address "the turn that raised request R" (a fourth
`RewindAnchor::RequestTurn { request_id }`). Captured here for completeness; deliberately **not**
implemented in the client now.

## 6. Durability / idempotency

- **Append-only seal (chosen: option b).** The journal stays append-only. On rewind the host records
  an append-only **seal record** in a `journal_seals` table `(stream, seal_cursor, retained_turns,
  epoch, recorded_unix)` where `seal_cursor` is the stream head at rewind time (`SessionStore::
  record_journal_seal`). The journal remains a complete audit log; `session_history` surfaces the
  latest seal as `JournalPageView::sealed_after` so a reconnecting client knows a rewind occurred and
  reconciles against the authoritative truncated `Snapshot` / `ConvView` rather than replaying the raw
  audit tail. (Per-turn dead-range *filtering* on the read path is intentionally deferred: conversation
  turn ordinals and per-`run_turn` journal segments are different granularities, and the authoritative
  truncated state is the engine snapshot, so the audit log is preserved intact and merely flagged.)
- **Epoch fencing.** `RewindTo` bumps the session `Epoch` (`Engine::rewind_to`, mirroring the suspend
  bump). The primary guard is interrupt-first ordering (the single-owner actor finishes the abandoned
  turn before truncating, so it emits no events after `Rewound`). As belt-and-suspenders for a *late*
  background-job completion from the abandoned tail, `Engine::apply_completions` drops any completion
  whose job is no longer in `waiting_for` (which the rewind cleared), and the durable activation path
  is fenced by the store `FenceToken` keyed to the bumped epoch. `Rewound { epoch }` publishes the new
  fence to clients.
- **Re-entrancy / interrupt-first.** `RewindTo` while a turn is live is well-defined: the actor
  cancels the live turn, lets it finalize as `Interrupted`, then applies the rewind at the boundary
  (so `&mut engine` is free and no abandoned-turn event races the truncation). A second `RewindTo`
  before the replayed `StartTurn` simply re-truncates to the new anchor.
- **Side effects (filesystem).** `RewindTo` ties into the §12 workspace checkpoints: `Engine::
  rewind_to` returns the `call_id`s of the tool calls in the sealed-off tail, and the host restores
  the **earliest** matching pre-mutation checkpoint (`CheckpointStore::restore`), which undoes every
  later mutation in the sealed range. Ordering: engine truncate + snapshot reconstruct + epoch bump +
  emit `Rewound`, then host journal seal + workspace rollback. The **no-checkpoint case** (a read-only
  rewound range that mutated nothing) leaves the filesystem as-is, which is correct.

## 7. Client-seam → NodeApi mapping

The desktop/TUI adapter stays a thin translation. The client funnels every rewind through one seam
(`ConversationOrchestrator`), so swapping the scripted `TurnController` for a `NodeApi` adapter
touches one place:

| Client seam (`daemon-app`) | NodeApi call |
| :--- | :--- |
| `DocumentStore::rewindToMessage(id)` (truncate inclusive, return text) | `ApiRequest::Rewind { point: RewindPoint { anchor: UserTurn(ordinal_of(id)), restore_workspace } }` |
| `DocumentStore::regenerateFromMessage(id)` (truncate reply, keep user) | `ApiRequest::Rewind { point: RewindPoint { anchor: ReplyAfter(ordinal_of(id)), restore_workspace } }` |
| `ConversationOrchestrator::rerun(text)` / `submit(text)` | `StartTurn { input: UserMsg::from(text) }` |
| `cancel()` before a rewind (interrupt-if-busy) | folded into `ControlApi::rewind` (interrupt-first) |
| (live) drop transcript tail on `Rewound` | `AgentEvent::Rewound { to_cursor }` |

`ordinal_of(messageId)` is the client→engine id resolution the adapter owns: the client's stable
message id maps to the engine turn ordinal / journal cursor carried alongside each streamed block.

## 8. Acceptance criteria

1. `RewindTo { UserTurn(o) }` then `StartTurn` re-runs from `o`; `session_history` flags the durable
   seal (`JournalPageView::sealed_after`) and the engine `ConvView` no longer contains turns at/after
   `o`.
2. `RewindTo` while a turn is live interrupts it first; no post-rewind event from the abandoned turn
   reaches a subscriber, and a late background completion from the abandoned tail is fenced.
3. A rewind across a turn with a workspace checkpoint rolls back the filesystem to the sealed-off
   range's earliest pre-mutation checkpoint.
4. `Rewound { to_cursor, epoch }` lets a live client converge on the engine's truncated state without
   a full `session_history` re-read.
5. Idempotent on `request_id`; a duplicate `RewindTo` re-truncates to the same anchor.
6. Foreign ACP sessions advertise `SessionInfo::rewindable = false` and a `RewindTo` submitted to a
   foreign agent is dropped (never faked).
7. Non-resident durable sessions return `Unsupported` until the host can re-incarnate, truncate, and
   reseal them without a live actor.

## 9. Foreign ACP sessions are non-rewindable

A foreign agent reached over the **Agent Client Protocol** (the `daemon-acp` adapter, presented as an
ordinary `UnitKind::Engine` managed unit) is **not rewindable**, by design. ACP (v0.15.0) has no
truncate-at-anchor primitive, and the foreign agent — not the daemon — owns its conversation state,
so the daemon cannot make it forget post-anchor turns:

- `session/load` **replays the full history** (no truncation).
- `session/fork` is `#[cfg(feature = "unstable_session_fork")]` and forks the *whole* context.
- `session/resume` does not replay or truncate.
- `session/cancel` only interrupts the current turn.

Rather than fake a rewind the foreign agent cannot honor, the daemon surfaces the limitation:

- The `AgentSession::rewindable()` capability returns `false` for `AcpSession` (and `true` for the
  in-tree daemon-core engine), queryable on a managed unit via `AgentUnit::rewindable()`.
- `SessionInfo::rewindable` is `false` for foreign-backed sessions (`true` for daemon-core), so a
  GUI/TUI **hides** the rewind affordance for ACP agents.
- Defensively, if a `RewindTo` reaches the ACP adapter anyway it is logged and dropped (no partial /
  faked rewind), and a `RewindTo` to a session that is not a live daemon-core session errors.
