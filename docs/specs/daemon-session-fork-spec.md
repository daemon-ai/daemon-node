# Quiescent-boundary session fork — design note (spec only, no code)

Status: DESIGN NOTE (engine-determinism track, Item 5). No implementation on this branch.

## What a fork is

A fork creates a new session whose conversation history is a copy of an existing session's durable
state at a chosen boundary, with its own identity and an independent future. It is a
**conversation fork**: the typed `Conversation` (and the composition that renders it, see the
composition-generation note) is what is copied. It is explicitly NOT a **workspace checkpoint** —
files, processes, and tool working state are not snapshotted or duplicated by a fork. The two
concepts compose (a fork MAY be paired with a workspace checkpoint by the operator) but are
separate operations with separate guarantees, and conflating them is the failure mode this note
exists to prevent: a forked child sharing the parent's live workspace is a *feature* (review a
divergent plan against the same tree) exactly as long as nobody pretends the files were versioned.

## The quiescent boundary

A fork is only well-defined at a **quiescent snapshot boundary**. Concretely, the source
`Snapshot` (`crates/engine/daemon-core/src/snapshot.rs`) must satisfy:

- **`waiting_for` is empty.** Outstanding background jobs are owned by the parent's incarnation;
  a fork taken mid-wait would either duplicate the job's eventual result into two sessions or
  strand the child waiting for a job that will report to the parent. Forbidden.
- **`pending_approvals` is empty.** A parked approval is a promise between the *parent's* operator
  and the *parent's* tool call, bound (post-A2) to a path-state or command fingerprint. Cloning it
  would let one human decision authorize two executions. Forbidden.
- **`references` is not carried.** `References.children`, `References.processes`, and
  `References.tools` are handles to live resources owned by the parent; the child starts with
  `References::default()`. Delegated children, OS processes, and tool state keys never follow a
  fork. (This is the workspace-checkpoint line again, stated in snapshot terms.)
- **No turn in flight.** Fork is a host operation on a `Ready`/suspended-at-boundary session, gated
  by the same activation-lease fencing as any other snapshot read; forking an `Active` session
  requires waiting for (or interrupting to) the next boundary.

What IS copied: `conversation`, `composed_prompt`/`composed_model` (byte-identical — the child's
first request must hit the same provider prefix cache the parent warmed), and the durable
engine-native cadence counters (`iters_since_skill` etc.) — the child continues the conversation
as if it were the parent. What is reset: `session_id` (fresh), `epoch` (starts at the child's own
0), `waiting_for`/`pending_approvals` (empty by precondition), `references` (empty), and
session-scoped approval memory (`approved_fingerprints` does NOT copy — a permanent allow is a
per-session grant; the child's operator re-grants).

## Lineage

The store records `(parent_session, parent_epoch, child_session, forked_at, reason)` — the
`record_lineage` shape the core spec already sketches — so provenance is queryable in both
directions: "what did this child fork from" and "what forked off this parent". Lineage is
append-only metadata; it never influences engine behaviour. The child's snapshot additionally
carries its origin `(parent_session, parent_epoch)` inline (`#[serde(default)]`), so a snapshot
blob is self-describing even outside the store.

## Interaction with the engine-determinism items

- B1a: the child's first assembled request digests over the copied conversation + composed prompt;
  agreement with the parent's last digest inputs is the fork-correctness smoke test.
- The durable inbox (sibling note): splices are per-session and do NOT copy — a fork boundary is by
  definition after all consumed splices and before any pending ones (a non-empty pending inbox is
  another quiescence violation).
- Composition generations (sibling note): the child inherits the parent's generation by value.

## Known limitations / deferred

- Fork of a session with live LCM externalized payloads requires either sharing the
  content-addressed side-channel (safe: content-addressed, immutable) or copying it; sharing is
  the default, with GC refcounting deferred to implementation.
- Cross-partition forks and fork-with-workspace-checkpoint are compositions deferred until the
  base operation exists.
