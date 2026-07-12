# Swarm-training MVP — lane R2 ledger (peer-side round engine / checkpointing / gossip)

Wave-2 coordination record for lane **R2** (`swarm/r2`). Companion to the program ledger
[`swarm-mvp-ledger.md`](swarm-mvp-ledger.md) ("Merge 1 — frozen interfaces") and the Wave-1
runtime record [`swarm-ledger-r1.md`](swarm-ledger-r1.md). Read those first for the frozen seams
(`daemon-swarm-proto` API, `SwarmTransport`, `TrainerBackend`, worker protocol) and the
frozen-file rule. This file records what R2 builds on top of those seams, the new seams it exports
(frozen at Merge 2), and every `MERGE-2` marker Merge 2 must resolve.

## Base + branch

- **Branch:** `swarm/r2`, forked from `c1432fa` (`mirror(merge-1): freeze cross-lane interfaces`)
  on `integrations/swarm` — Merge 1, all Wave-1 lanes (P1/R1/E1) integrated.
- **Merge target:** `integrations/swarm` (disjoint file set → conflict-free by construction).
- **Owns (create / edit only within):** `crates/swarm/daemon-swarm-net`,
  `crates/swarm/daemon-swarm-run`, `crates/coprocessor/daemon-train-client`, and the `tests/`
  swarm e2e surface (`tests/daemon-swarm-e2e/`).

## FROZEN — do not touch (single-writer rule)

Root `Cargo.toml`, `deny.toml`, `flake.nix`; the Merge-1 frozen seams (`daemon-swarm-proto` API,
`SwarmTransport` traits, `TrainerBackend` trait, worker protocol message set). Extend the frozen
seams **additively only**. Other lanes' directories (coordinator/observe/proto/det-core/train-sdk/
train/guests/xtask) are out of bounds.

## Parallel-lane note (P2 coordinator)

Lane P2 builds the coordinator pure-tick state machine + deterministic assignment **in parallel**;
its API is **not** available to R2. The peer-side round engine here is built directly against the
frozen Wave-1 proto message types (`RoundOpen`/`Commitment`/`Attestation`/`StorageReceipt`/
`RoundRecord`/`Digest`/`Straggle`). Every place that would consume a real coordinator drives a
**TEST-ONLY scripted coordinator** fixture (a hardcoded/scripted message sequence over
`LoopbackGossip`), clearly marked `// MERGE-2: replace with daemon-swarm-coordinator tick loop`.
R2 does **not** build a general coordinator.

## Scope (this wave)

| Slice | Crate | Spec | TDD |
|---|---|---|---|
| Reusable control-plane `Deduper` (blake3) | `daemon-swarm-net` | §7.1 | NET-6 |
| Payload fetch fallback (backoff + alternate locators) | `daemon-swarm-net` | §7.1 | NET-4 |
| Reconverging `StubBackend` (round-base outer step) | `daemon-swarm-run` | §5.6, §6.4 | (round-loop oracle) |
| Peer-side `RoundEngine` (round protocol, barrier, stall ladder) | `daemon-swarm-run` | §6.4 | RUN-1..5, RUN-8 |
| Round-boundary checkpoint + desync replay resync | `daemon-swarm-run` | §9, §5.6 | RUN-6/7 subset |
| `harness` (peers + scripted coordinator) | `daemon-swarm-run` | §6.4 | (e2e support) |
| Stub e2e (N=3 × 20 rounds, stall+catchup, deterministic) | `tests/daemon-swarm-e2e` | §6.4, §19.5 | P0 milestone (Merge 2) |

## Seams R2 exports (freeze at Merge 2)

- **`RoundEngine`** (`daemon_swarm_run::engine`) — an async peer-side round state machine
  constructed over `ControlPlane` + `PayloadStore` + `TrainerBackend` + the node ed25519
  `SigningKey`, plus an `EngineConfig` and an event sink. It drives one peer through rounds and
  emits an `EngineEvent` outcome stream. Signature (verbatim below).
- **Checkpoint types** (`daemon_swarm_run::checkpoint`) — `CheckpointManifest { round, blake3,
  digest }`, `Checkpointer`, and the `resync` replay hook.
- **The e2e harness shape** (`daemon_swarm_run::harness`, `feature = "harness"`) — the in-process
  peer set + the `ScriptedCoordinator` fixture. This is the shape the Merge-2 P0 milestone test
  keeps, swapping the scripted coordinator for the real `daemon-swarm-coordinator` tick loop.

### `RoundEngine` API (the seam to watch)

```rust
pub struct EngineConfig {
    pub run: RunId,
    pub roster: Vec<PeerId>,        // frozen for the epoch; sorted internally (I3 order)
    pub witnesses: Vec<PeerId>,     // whose Attestations count (§6.4); default = roster
    pub steps_per_round: u32,
    pub micro_batch: u32,
    pub stall_rounds_max: u32,      // §6.4 rung 2 budget (default 2)
    pub checkpoint_every_rounds: u32,   // 0 = round-boundary checkpoints off
    pub version: SwarmProtoVersion,
}

pub enum EngineEvent {
    Committed { round: RoundId, hash: ContentHash },
    Attested  { round: RoundId, root: Root, count: u32 },
    RoundComplete { round: RoundId, digest: StateDigest },
    Straggling { round: RoundId, status: StraggleStatus },
    CaughtUp { round: RoundId, digest: StateDigest },
    Checkpointed { round: RoundId, manifest: CheckpointManifest },
    Left { round: RoundId, reason: String },
}

impl<C: ControlPlane, P: PayloadStore, B: TrainerBackend> RoundEngine<C, P, B> {
    pub fn new(control: Arc<C>, store: Arc<P>, backend: B, key: SigningKey,
               corpus: Arc<Corpus>, cfg: EngineConfig,
               events: mpsc::UnboundedSender<EngineEvent>) -> Self;
    pub async fn run(&mut self) -> Result<RunOutcome, SwarmRunError>;
}

pub enum RunOutcome { Finished { last_round: Option<RoundId> }, LeftForEpoch { round: RoundId } }
```

The engine subscribes to the control plane at construction. `run()` is the message-driven loop:
`RoundOpen(r)` → derive interval → train (`train_step` × micro-batches → `inner_update` per inner
step → `make_update`) → PUT payload → publish signed `Commitment`; prefetch + blake3-verify peers'
payloads as `Commitment`s arrive (witnesses also publish `Attestation` over their fetch-verified
set); `RoundRecord(r)` is enqueued and ingest advances **in strict ascending round order** — the
first ingestible round is the **barrier** (verify the committed set against the record root, stage
in record order (node-pubkey bytes, I3), `ingest` → `Digest`, then a cadence checkpoint); a round
whose set is not yet fetchable blocks every later round behind it (the stall ladder), so no outer
step is applied out of order. The loop terminates when the control plane closes (`Finished`) or on
stall-budget exhaustion (`LeftForEpoch`); the harness tears peers down after the last round.

Notes on the delivered API vs the initial sketch: fetch retry/backoff (NET-4 `RetryPolicy` /
`fetch_with_fallback`) lives at the **net** layer for the real network payload plane — the engine
uses single-attempt `get` at the barrier and treats *cross-round* re-fetch as the stall-ladder
retry, which keeps the fault-injected stall test deterministic. Digest sampling params (§5.6) are
the host/backend's concern (the `TrainerBackend::ingest` return value), not `EngineConfig`.

## Design decisions (not obvious from the code)

- **`StubBackend` now reconverges (DiLoCo-shape outer step).** The Wave-1 `StubBackend::ingest`
  folded staged payloads into *current* params, so two peers that trained on different windows
  produced different post-ingest digests — the round-loop "equal digest each round" property
  (§5.6) could never hold. R2 refactors the stub to the spec's agree-path shape: a `base` snapshot
  (the consensus round base, ABI §5.9) is the outer-step anchor; `make_update` emits this peer's
  delta relative to `base`; `ingest` sets `params = base ⊕ orderedFold(staged)` then re-snapshots
  `base`. Because `base` is equal across peers post-ingest and the committed set (record order) is
  equal, the digest is equal every round — while local training still legitimately diverges
  `params` between barriers. Ordering-sensitivity and determinism (the `record_ordering.rs`
  cross-crate I3 test) are preserved. The `TrainerBackend` **trait** is untouched (frozen).
- **Barrier (I2) via a single-task engine.** The engine owns `&mut backend` and processes control
  messages sequentially, so the first `train_step` of round r+1 cannot begin until `ingest(r)`
  returns — the barrier is structural, not advisory. `RoundOpen(r+1)` ships with `RoundRecord(r)`
  and is queued behind the record in the subscription, so it is handled strictly after ingest.
- **Fetch/compute overlap is swarm-level + reactive.** Overlap is realized by the async runtime
  interleaving peers' tasks: while peer A trains round r, peers B/C are committing and A prefetches
  their payloads reactively as `Commitment`s arrive (verified on receipt, cached), so the barrier
  usually finds the set already local. A dedicated per-peer concurrent fetch task (true
  fs/network parallelism *inside* one peer while it computes) is deferred — with the fast
  `FsPayloadStore` it buys nothing, and it would add nondeterminism to the digest transcript.
  `// MERGE-2`: revisit a spawned fetch task once the real iroh/r2 payload plane makes in-peer
  fetch latency material (Wave 3).
- **Local equal-split assignment.** The engine derives a peer's `BatchInterval` from
  `(RoundOpen.batch, roster index)` with a deterministic contiguous equal split.
  `// MERGE-2`: replace with `daemon-swarm-coordinator`'s throughput-weighted deterministic
  assignment (§6.3, PROTO-8); the split site is isolated in `engine::assignment`.
- **Desync detection input.** The record does not carry a consensus digest (each peer emits its
  own `Digest`). R2's resync hook takes an explicit expected/quorum digest (supplied by the
  harness/observer); the replay itself is spec-faithful (checkpoint reload + record/payload replay,
  I1). `// MERGE-2`: wire the quorum-digest source from `daemon-swarm-observe` / the coordinator's
  digest tally.

## MERGE-2 marker sites (search `MERGE-2` in the tree)

| Site | What Merge 2 must do |
|---|---|
| `daemon-swarm-run/src/harness.rs` `ScriptedCoordinator` | swap for the real `daemon-swarm-coordinator` tick loop; the harness peer-set shape stays |
| `tests/daemon-swarm-e2e/tests/*.rs` | swap scripted coordinator → real tick loop (becomes the P0 milestone test) |
| `daemon-swarm-run/src/engine.rs` `assignment` | replace equal-split with coordinator deterministic assignment (§6.3) |
| `daemon-swarm-run/src/engine.rs` fetch overlap note | consider a spawned in-peer fetch task once the network payload plane lands |
| `daemon-swarm-run/src/checkpoint.rs` resync quorum digest | wire the consensus-digest source (observe/coordinator tally) |

## Things Merge 2 / later waves must watch for

- **`tests/*` is a workspace-member glob**, so `tests/daemon-swarm-e2e/` is picked up with **no**
  root `Cargo.toml` edit — the e2e landed as its own crate (the preferred R1 plan location), not as
  a fallback integration test inside `daemon-swarm-run`. It depends on `daemon-swarm-run` with the
  `harness` feature.
- **The `harness` feature** on `daemon-swarm-run` gates the reusable in-process peer harness +
  scripted coordinator (test-fixture code kept out of the default build). Additive; no frozen-file
  change. The e2e crate and any Merge-2 milestone test enable it.
- **Payload fetch fallback (NET-4)** is a store-list + bounded backoff (`fetch_with_fallback`);
  only the `File`/`Fs` plane exists, so the test uses a second `FsPayloadStore` as the fallback
  source. Alternate locators from the `Commitment` (`BlobTicket`) await the iroh-blobs plane.
- **`Deduper` (NET-6)** is now a reusable type; `LoopbackGossip` composes it. Any future control
  plane (WS, iroh gossip) reuses the same content-hash dedupe rather than re-implementing it.
- **Additive-only extension** of the frozen seams remains the rule; the `RoundEngine` /
  checkpoint / harness seams above freeze at Merge 2.
