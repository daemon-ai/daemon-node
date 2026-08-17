# Durable inbox lifecycle — design note (spec only, no code)

Status: DESIGN NOTE (engine-determinism track, Item 5). No implementation on this branch.

## Problem

An inbound session message (start-turn text, steer, observation) is currently accepted into the
live projection before anything durable records it. `MergedLog`
(`crates/substrate/daemon-host/src/node_api/internals.rs`) is an **in-memory ring**: it stamps a
`seq`, retains history for paging, and fans entries out to subscribers — but a node restart loses
every entry that had not yet been folded into an engine `Snapshot` by a turn boundary. The gap is
the window between "the client saw its message accepted" and "a snapshot captured its effect": a
crash inside that window silently drops accepted input.

dsh closes the same gap with log-splice-before-projection: the inbound event is appended to the
durable log first, acknowledged second, and projected third; on resume the projection is rebuilt
by replaying the durable log after the last checkpoint. This note ports that shape onto the
daemon-node substrate.

**Non-assumption (review finding, binding):** `SessionLogEntry` does NOT imply durability. The
type is shared by the live ring and (future) durable splices, but only an explicit `SessionStore`
write is durable. Any implementation that "already appends to the session log" has proven
nothing.

## The durable transaction

Ordering is the contract; each step is observable only after the previous one committed:

1. **Append durable splice.** A new `SessionStore` operation appends an *inbox splice event*
   `(session, splice_seq, claim_state, payload, received_at)` to the session's durable row family,
   in the same store (and transactional domain) as the snapshot blobs
   (`crates/substrate/daemon-store/src/lib.rs`, `trait SessionStore`). `splice_seq` is a
   store-assigned monotonic sequence per session — the durable cousin of the ring's `seq`, but
   never reusing it (the ring renumbers on restart; the splice never does).
2. **Acknowledge acceptance.** Only after the splice commits does the node acknowledge the client
   request. An acknowledged message is therefore durable by definition.
3. **Update the live projection.** The `MergedLog` append + broadcast (the current behaviour)
   happens last, stamped with the splice's identity so a subscriber can correlate live entries
   with durable ones.

On restart: rehydrate the snapshot, then **replay durable splice events after the checkpoint
cursor** — the snapshot records the highest `splice_seq` it has consumed (a new snapshot field,
`#[serde(default)]` per existing style), and every splice above it is re-enqueued into the
engine's inbox before the session accepts new input. The live ring is rebuilt from the replayed
tail; its `seq` values are fresh (subscribers reconnect and re-page; they never persist ring
seqs).

## Claim semantics

Each splice event carries a claim state so replay is idempotent and observable:

- `Pending` — appended, acknowledged, not yet consumed by a turn.
- `Claimed { epoch }` — an engine incarnation (fenced by `Snapshot.epoch`) has taken the splice
  into a turn that has not yet reached a snapshot boundary. A crash here reverts the claim: on
  replay, a `Claimed` splice whose epoch is older than the rehydrated snapshot's epoch is treated
  as `Pending` again (the turn it fed never became durable, so re-consuming it is exactly-once
  *with respect to durable effects*).
- `Consumed { snapshot_epoch }` — a snapshot at or after `snapshot_epoch` captured the splice's
  effect; replay skips it. Set transactionally with the snapshot write, never separately.

Per message kind:

- **StartTurn**: one splice per user message; claimed by the turn that folds it into
  `Conversation.turns`; consumed at the turn's first snapshot boundary.
- **Steer**: spliced like StartTurn but consumed by whichever turn drains it (the engine's
  existing steer-drain points); a steer that arrives while no turn runs replays as a queued steer.
- **Observe** (read-only attach): NOT spliced. Observation has no durable effect and must not
  serialize behind the inbox; it rides the live ring only. This is the explicit line between the
  durable inbox (state-changing intents) and the projection (everything a client renders).

## Interaction with the existing machinery

- `Snapshot.epoch` already fences stale incarnations; claims reuse it rather than inventing a
  second fencing domain.
- The activation lease (`SessionStore::acquire`/fencing token) already guarantees a single
  incarnation appends splices for a session, so `splice_seq` needs no cross-writer coordination.
- The engine is unaware of all of this (it already consumes an inbox); the port lives entirely in
  the host/store layers, preserving the "node decides, engine is substrate-blind" layering.

## Known limitations / deferred

- Exactly-once is scoped to durable effects; a turn's *external* side effects (a tool that already
  ran) are governed by the approval/idempotency machinery, not the inbox.
- Batching multiple splices into one store transaction (throughput) is an optimization left to
  implementation; the ordering contract above is what may not be weakened.
- Cross-node inbox handoff (session migration) rides the sync protocol and is out of scope here.
