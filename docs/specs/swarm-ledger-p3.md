# Swarm-training MVP — lane P3 ledger (consensus / observability)

Wave-3 (final wave) working record for lane **P3** (`swarm/p3`). Scope is the observer / replay-oracle
crate (`daemon-swarm-observe`), plus additive extensions on the frozen consensus surfaces this
program froze at Merge 1 and Merge 2: the content-addressed `record-set.cbor` object codec + the
root-only attestation coverage path (proto + coordinator), an additive `Join` envelope-hash carrier,
and a warmup early-exit signalled through an additive `Heartbeat` readiness flag.

Read first, in order:
[`swarm-mvp-ledger.md`](swarm-mvp-ledger.md) (**Merge 1 AND Merge 2 frozen-interface sections** —
everything there is FROZEN, extend additively only), [`swarm-ledger-p2.md`](swarm-ledger-p2.md)
(Merge-2 carried notes this lane closes out), `swarm-training-spec.md` §6.4 (round protocol,
record-set objects, attestation roots), §10.1 (`daemon-swarm-observe`'s role), §14 (observability),
and `swarm-training-tdd.md` §3.9 (OBS) + PROTO-20 (replay oracle).

## Base + branch

- **Repo:** `daemon-node` (Rust backend submodule; standalone checkout).
- **Base commit:** `39c0ebd` (`mirror(merge-2): P0 milestone — freeze Wave-2 interfaces`) — the
  Merge-2 tip on `integrations/swarm` (real coordinator `tick` drives the e2e; replay verification).
- **Branch:** `swarm/p3`, forked from `39c0ebd`. Integrates back into `integrations/swarm` at Merge 3.
- **Worktree:** `/home/j/experiments/daemon-worktree/swarm-proto` (isolated; never touches the
  read-only checkout at `/home/j/experiments/daemon`).

## Owned files (this lane, disjoint by construction)

- `crates/contracts/daemon-swarm-proto/` (+ its `daemon-swarm.cddl`, dev-deps).
- `crates/swarm/daemon-swarm-coordinator/`.
- `crates/swarm/daemon-swarm-observe/`.

FROZEN / off-limits: root `Cargo.toml`, `deny.toml`, `flake.nix`; every other lane's directory
(`crates/swarm/daemon-swarm-{net,run}`, `crates/coprocessor/*`, `crates/contracts/{det-core,
daemon-train-sdk}`, `guests/`, `xtask/`, `bins/`, `tests/`).

## Scope this wave (lane P3, Wave 3)

| Area | Spec / TDD grounding | Deliverable |
|---|---|---|
| Observer / replay-oracle crate | §10.1, §14; TDD §3.9, PROTO-20 | `daemon-swarm-observe`: message log (writer/reader), replay oracle, digest tally / `DesyncVerdict`, `RunHealth` |
| Record-set object codec | §6.4, §11.3; TDD PROTO-5/RUN-2 | proto `RecordSet` (content-addressed committed-set object): canonical CBOR of sorted entries, blake3 = locator hash, commitment/membership |
| Root-only attestation coverage | §6.4 I6 | coordinator commit rule: root-agreement across the witness quorum pins coverage without every witness inlining its set |
| Envelope-hash admission (additive) | §6.1/§6.5; TDD PROTO-12 | `EnvelopeHashMismatch` enforcement in `admit` (+ the `Join` wire carrier — see the constraint note) |
| Warmup early-exit (additive) | §6.2/§6.5 | exit `Warmup` when all admitted members signal readiness, via an additive `Heartbeat` readiness flag |

## The frozen-surface constraint that shaped this lane (important)

Lane R's `daemon-swarm-run` (off-limits to P3) constructs several of this lane's public types with
**all-fields struct literals** that P3 may not edit:

- `harness.rs` builds `Join { run_id, iroh_id, class, capabilities }` (4-field literal).
- `harness.rs` builds `RunConfig { … }` and `CoordinatorParams { … }` as full literals.

Consequently, **adding a field to `Join`, `RunConfig`, or `CoordinatorParams` breaks the workspace
build** (`cargo test --workspace` / `cargo clippy --workspace --all-targets` compile
`daemon-swarm-run`'s `harness` via `cfg(test)` and via the e2e crate's `harness` feature). Rust
struct literals are exhaustive, so there is no `#[serde(default)]`/`#[non_exhaustive]` escape.

By contrast lane R uses `CoordinatorState::new(config, seed, now)` (a constructor, not a literal) and
**never constructs `Member` or the swarm `Heartbeat`**. So P3 may additively extend
`CoordinatorState`, `Member`, and `Heartbeat`, but **not** `Join`, `RunConfig`, or
`CoordinatorParams`. This is the disjoint-merge guarantee working as designed: a field on a shared
type consumed by another lane is a coordinated change, not a lane-isolated one.

**Resolution taken:**

- **Warmup early-exit** rides an additive `Heartbeat.ready` flag (`Heartbeat` is not literal-built by
  any other lane) → lands fully in P3.
- **Envelope-hash admission** is enforced in `admit` (the `EnvelopeHashMismatch` reason +
  `JoinCandidate::asserted_hash` have existed since P2, tick passes it through). The **`Join` wire
  carrier** (`envelope_hash: Option<Hash>`) is a **Merge-3-coordinated** additive change: P3 ships the
  fully-tested enforcement, and lane R adds `envelope_hash: None` to its one `Join` literal in the
  same integration (a mechanical one-line edit). P3 keeps `Join`/its CDDL rule in lock-step with the
  struct (no desync) and records the exact CDDL delta below for the integrator to apply.

## Design decisions (rationale for choices the spec left to the lane)

1. **`RecordSet` lives in `daemon-swarm-proto`, wasm-clean.** The `record-set.cbor` object (§11.3) is
   the content-addressed opening of a `RoundRecord`'s set commitment; a replaying peer (or the
   observe oracle) fetches it, verifies it against the record's root, and stages it (§6.4 barrier).
   Placing the codec in proto keeps one canonical encoder and lets the coordinator, peers, and the
   observer all consume it without a cross-crate fork. `content_hash()` = blake3 of the canonical
   CBOR (the locator hash, per the lane brief); `commitment()` = the same `commit_set` root the
   `RoundRecord` signs, so `verify_against(&record.set)` is exact (root + count).

2. **Root-only coverage relaxes the inline requirement, not the security.** The Merge-2 commit rule
   required a witness **quorum of inline sets** each covering `(peer, hash)`. Wave 3 adds a
   scale-invariant path: a witness may attest a **bare root** (inline omitted), and when a quorum of
   witnesses agree on the same root `R`, membership of `(peer, hash)` in `R` is pinned by **either** a
   `StorageReceipt` (the coordinator-as-storage-client availability evidence, §6.4 I6) **or** a
   single inline opening that reconstructs `R`. A full opening whose blake3-merkle root equals the
   quorum-agreed root *is* the committed set, so membership stays exact — no probabilistic input, no
   trust in an un-opened root. The pure-root, no-opening, no-receipt case is (correctly) not
   admissible: some opening or receipt must pin the element (this is exactly "StorageReceipts + root
   equality across the witness quorum", spec §6.4). `SetCommitment::verify_membership` (frozen since
   Merge 1) remains the O(log n) proof path a future coordinator that *holds* proofs would use.

3. **`daemon-swarm-observe` consumes only signed messages + published objects.** No privileged
   coordinator state: the replay oracle rebuilds genesis from `(envelope, params)` and folds a `tick`
   `Input` trace, exactly like the Merge-2 harness `CoordinatorReplay` — observe is its **library**
   form (TDD PROTO-20). The message log stores the signed-message subset (append-only, canonical-CBOR
   framed); the clock trace is the driver's thin sidecar (the harness already records the exact input
   sequence). On the **event-driven happy path** (commit + evidence finalize a round with zero
   clocks — the Merge-2 P0 finding) the replay reproduces every `RoundRecord` from the messages
   alone; timeout/straggler rounds additionally need the recorded clocks (carried in the `Input`
   trace, not the message log).

4. **`daemon-swarm-observe` uses std freely (node-side tool).** The scaffold set `thiserror` + `serde`
   and is **not** on the `wasm32` substrate path (§10.1 lists it as a node-side event-sourced log; it
   is never linked into the coordinator DO). The message-log framing uses `std::io::{Read, Write}`.
   Kept as scaffolded — proto and the coordinator stay wasm-clean (COORD-3); observe does not.

5. **Warmup early-exit is additive and opt-in.** `Heartbeat` gains an optional `ready` flag; a member
   that heartbeats `ready = Some(true)` during `Warmup` is recorded ready (`Member.warmup_ready`).
   When every healthy member is ready, `tick` opens round 0 immediately instead of waiting for the
   `warmup` timeout. Absent readiness signals (every existing caller, incl. lane R's harness and the
   e2e) → the timeout path is unchanged (back-compat). The `WaitingForMembers → Warmup` transition
   stays clock-driven (changing it would break `proto2`'s pre-clock assertion); only the
   `Warmup → RoundTrain` exit gains the early path.

## Additive extensions made (freeze at Merge 3) — exact

### proto (`daemon-swarm-proto`)

- **New module `record_set`** (new type + functions; no change to any frozen message):
  `RecordSet { entries: Vec<RecordEntry> }`, `RecordSet::new(iter)` (sorts by peer bytes then
  hash+size, dedups), `to_canonical_vec`, `from_canonical_slice`, `content_hash() -> Hash` (blake3 of
  the canonical CBOR), `commitment() -> SetCommitment`, `verify_against(&SetCommitment)`,
  `entries()`, `is_empty`/`len`. Re-exported at the crate root.
  - **CDDL, additive (new rule):** `record-set = { "entries": [* record-entry] }`.

- **`Heartbeat` — additive optional field:**
  `pub ready: Option<bool>` (`#[serde(default, skip_serializing_if = "Option::is_none")]`). Absent on
  the wire for legacy senders (back-compat); `Some(true)` = model-ready during `Warmup`.
  - **CDDL, additive:** `heartbeat = { "round": round, ? "ready": bool }`.

- **`Join` — additive optional field (MERGE-3-COORDINATED, not yet applied — see the constraint
  note):** the intended addition is `pub envelope_hash: Option<Hash>`
  (`#[serde(default, skip_serializing_if = "Option::is_none")]`); CDDL
  `join = { …, ? "envelope_hash": hash }`. **Not applied on `swarm/p3`** because it would break lane
  R's frozen 4-field `Join` literal; the integrator applies it together with lane R's one-line
  `envelope_hash: None` edit at Merge 3. The enforcement path is already complete + tested (below).

### coordinator (`daemon-swarm-coordinator`)

- **`commit::has_evidence` — additive root-only path** (decision 2). New helper
  `commit::quorum_root(rs) -> Option<Root>` (a root attested by ≥ `witness_quorum` witnesses).
- **`Member` — additive field** `pub warmup_ready: bool` (set `false` by `Member::joining`; reset on
  `Warmup` entry; set by a ready heartbeat).
- **`CoordinatorState` — no new field** (readiness lives on `Member`; `::new` unchanged, so lane R's
  constructor call is untouched).
- **`tick`** — `on_heartbeat` records warmup readiness; `drive_time`'s `Warmup` arm opens round 0
  early when all healthy members are ready. `admit` envelope-hash enforcement unchanged (already
  wired for `asserted_hash`); tick still passes `None` until the `Join` carrier lands (Merge 3).

## Seams this lane exports (freeze at Merge 3)

- **observe:** `MessageLog` (append/iter/`by_round`/`by_kind`/`by_round_kind`/`write_to`/`read_from`/
  `replay_inputs`), `MessageKind`, `replay(envelope, params, impl Iterator<Item = Input>) ->
  Result<ReplayReport, ReplayDivergence>`, `ReplayReport`, `ReplayDivergence`, `genesis_seed`,
  `digest_tally` + `DesyncVerdict`, `RunHealth` + `RoundHealth` (`RunHealth::from_log`), `ObserveError`.
- **proto:** `RecordSet` codec (record-set object), `Heartbeat.ready`.
- **coordinator:** root-only commit-rule semantics (`commit::has_evidence`, `commit::quorum_root`),
  `Member.warmup_ready` + warmup early-exit.

## Planned commit slices (each green per the gates; TDD tight test+impl slices)

1. `mirror(P3): ledger` — this file.
2. `feat(swarm-proto): record-set object codec + membership helpers (green)`.
3. `feat(swarm-coordinator): root-only attestation coverage in the commit rule (green)`.
4. `feat(swarm-proto): additive Heartbeat readiness flag (green)`.
5. `feat(swarm-coordinator): warmup early-exit on peer readiness (green)`.
6. `feat(swarm-coordinator): envelope-hash admission enforcement tests (green)`.
7. `feat(swarm-observe): message log + replay oracle + digest tally + run health (green)`.
8. `mirror(P3): ledger — Merge-3 seams + results` (final, after the full-workspace gates).

Every commit passes `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`, `cargo build --target wasm32-unknown-unknown -p daemon-swarm-proto`
(**and** `-p daemon-swarm-coordinator`), and `typos docs/specs`.

## Wave-3 results — frozen seams (Merge 3)

Landed on `swarm/p3` (base `39c0ebd`). Commits (oldest → newest):

| Commit | Subject |
|---|---|
| `3325061` | `mirror(P3): ledger` |
| `e290419` | `feat(swarm-proto): record-set object codec + membership helpers (green)` |
| `1d333d0` | `feat(swarm-coordinator): root-only attestation coverage in the commit rule (green)` |
| `37e1d4f` | `feat(swarm-proto): additive Heartbeat readiness flag (green)` |
| `bba2462` | `feat(swarm-coordinator): warmup early-exit on peer readiness (green)` |
| `657fa7d` | `feat(swarm-coordinator): envelope-hash admission enforcement tests (green)` |
| `9915370` | `feat(swarm-observe): message log + replay oracle + digest tally + run health (green)` |

**observe seam** (`daemon_swarm_observe`; node-side std tool):

```rust
// message log (append-only, canonical-CBOR length-framed, (round, kind) indexed)
pub struct MessageLog;                        // new/run_id/append/len/is_empty/iter/entries/rounds
pub enum  MessageKind { RoundOpen, Commitment, Attestation, StorageReceipt,
                        RoundRecord, Digest, Straggle, Join, Heartbeat }
impl MessageLog {
    pub fn by_round(&self, round: u64)                 -> impl Iterator<Item = &SignedMessage>;
    pub fn by_kind(&self, kind: MessageKind)           -> impl Iterator<Item = &SignedMessage>;
    pub fn by_round_kind(&self, round: u64, kind: MessageKind) -> impl Iterator<Item = &SignedMessage>;
    pub fn replay_inputs(&self)                        -> impl Iterator<Item = Input> + '_;
    pub fn write_to(&self, w: &mut impl std::io::Write) -> Result<(), ObserveError>;
    pub fn read_from(r: &mut impl std::io::Read)        -> Result<Self, ObserveError>;
}

// replay oracle (PROTO-20 as a library)
pub fn genesis_seed(env: &Envelope) -> Result<Seed, ObserveError>;
pub fn replay(env: &Envelope, params: CoordinatorParams, inputs: impl Iterator<Item = Input>)
    -> Result<ReplayReport, ReplayError>;
pub struct ReplayReport   { pub records: Vec<RoundRecord>, pub rounds_verified: u64,
                            pub final_state_hash: Hash }
pub struct ReplayDivergence{ pub round: u64, pub recorded: RoundRecord,
                            pub rederived: Option<RoundRecord>, pub detail: String }
pub enum   ReplayError    { Setup(ObserveError), Diverged(Box<ReplayDivergence>) }

// desync detection (the observe-driven trigger R3 consumes)
pub fn digest_tally(round: u64, reports: impl IntoIterator<Item = (PeerId, StateDigest)>, quorum: u32)
    -> DesyncVerdict;
pub fn desync::digest_tally_from_log(log: &MessageLog, round: u64, quorum: u32) -> DesyncVerdict;
pub struct DesyncVerdict  { pub round: u64, pub quorum_digest: Option<StateDigest>,
                            pub outliers: Vec<PeerId>, pub reporters: u32, pub agreed: bool }
                            // + is_desync()

// run health (plain serializable projection for CLI/UX)
pub struct RunHealth   { pub run_id: String, pub rounds: Vec<RoundHealth> }  // RunHealth::from_log
pub struct RoundHealth { pub round: u64, pub committed: u32, pub attested_coverage: u32,
                         pub stragglers: Vec<PeerId>, pub drops: Vec<PeerId>,
                         pub digest_reporters: u32, pub digest_agreed: bool,
                         pub duration_ticks: u64, pub finalized: bool }
```

> **Deviation from the brief's literal `replay(...) -> Result<ReplayReport, ReplayDivergence>`:** the
> concrete return is `Result<ReplayReport, ReplayError>`, where `ReplayError::Diverged(Box<ReplayDivergence>)`
> carries the pinpointed first divergence and `ReplayError::Setup(ObserveError)` carries an
> envelope/genesis setup failure (a real, non-divergence error path). The `ReplayDivergence` payload
> is exactly as specified (round, recorded-vs-rederived output). Boxed to satisfy
> `clippy::result_large_err`.

**proto seam** (`daemon_swarm_proto`, wasm32-clean): `RecordSet` (`new`/`to_canonical_vec`/
`from_canonical_slice`/`content_hash`/`commitment`/`verify_against`/`entries`/`len`/`is_empty`);
`Heartbeat.ready: Option<bool>`. **CDDL additions:** `record-set = { "entries": [* record-entry] }`;
`heartbeat = { "round": round, ? "ready": bool }`.

**coordinator seam** (`daemon_swarm_coordinator`, wasm32-clean): `commit::has_evidence` (now with the
root-only path) + `commit::quorum_root`; `Member.warmup_ready: bool` + warmup early-exit in `tick`.

**Warmup early-exit landed** (not documented-out): `Heartbeat` is not literal-constructed by any other
lane, so the additive `ready` flag + `Member.warmup_ready` + the event-driven exit are complete and
tested. The `WaitingForMembers → Warmup` transition stays clock-driven (an event-driven variant would
break `proto2`'s pre-clock assertion).

**Gates** (all green, run via `nix develop --command …` from the worktree root): `cargo fmt --check`;
`cargo clippy --workspace --all-targets -- -D warnings`; `cargo build --target wasm32-unknown-unknown
-p daemon-swarm-proto` **and** `-p daemon-swarm-coordinator`; `typos docs/specs`. `cargo test
--workspace` is green **except** the two pre-existing `daemon-conformance` detached-delegation flakes
(`detached_fanout_materializes_distinct_children`, `detached_notice_reaches_a_parked_durable_parent`)
— outside this lane, untouched, and **green in isolation** (`cargo test -p daemon-conformance --lib
detached_delegation` → 5/5). Test counts (this lane): proto record-set 5 + heartbeat-cddl 1 (+ prior
suites); coordinator root-only 3 + warmup 2 + envelope-hash 2 (+ prior 27→**34** commit/tick/admission
tests); observe **6** (`log_roundtrip_canonical`, `replay_matches_live_run`,
`replay_detects_tampered_record`, `digest_quorum_flags_outlier`, `run_health_projects_per_round_facts`,
+ error-render unit).

## Notes for Merge 3 integration (what to watch)

- **`Join.envelope_hash` is the one deferred, coordinated change.** Apply the additive field
  `pub envelope_hash: Option<Hash>` (`#[serde(default, skip_serializing_if = "Option::is_none")]`) to
  `daemon_swarm_proto::messages::Join`, the CDDL rule `join = { …, ? "envelope_hash": hash }`, and in
  the **same** integration commit fix lane R's single `Join { … }` literal in
  `daemon-swarm-run/src/harness.rs` (add `envelope_hash: None`). Then thread it through
  `tick::on_join` (`asserted_hash: j.envelope_hash.as_ref()`); the `admit` check +
  `AdmissionReject::EnvelopeHashMismatch` are already present and tested. This is the only P3 seam that
  could not land in isolation (the frozen lane-R literal), so it is left as an explicit integrator task
  rather than a broken workspace gate.
- **observe depends additively on `daemon-swarm-coordinator`** (path dep already in the frozen root
  `[workspace.dependencies]`; lane-owned `Cargo.toml` edit only — no root/`deny`/`flake` change, same
  pattern R2 used for the harness coordinator dep). Re-run `cargo deny check` at the merge is a no-op
  (no new third-party dep; `ciborium` dev-dep is already pinned).
- **`DesyncVerdict` is exported, not wired.** R3 consumes it at `daemon-swarm-run/src/checkpoint.rs`'s
  desync-trigger marker (the MERGE-2 note): fold the run's `Digest`s (or use `digest_tally_from_log`)
  and drive resync on `is_desync()`. P3 deliberately does **not** touch `daemon-swarm-run`.
- **Replay clock sidecar.** `replay` reproduces the coordinator from the recorded `tick` `Input` trace.
  The message log stores the signed-message subset; timeout/straggler rounds also need the recorded
  `Input::Clock`s (the driver's sidecar — the Merge-2 harness `CoordinatorReplay` already records the
  exact input sequence, so observe's `replay` is its library form). The happy path is clock-free thanks
  to event-driven finalize + the new warmup readiness exit.
- **State shape changed additively** (`Member.warmup_ready`). `CoordinatorState` CBOR gained a field;
  same-code replay/round-trip is unaffected (no golden state-byte vectors exist). Any *cross-wave*
  persisted `CoordinatorState` snapshot from before Merge 3 would need `#[serde(default)]` if replayed
  by newer code — not a concern within a single run.
- **`RecordSet` vs `RoundRecord.set_locator`.** `tick` still emits a `StoreKey` locator
  (`runs/<run>/rounds/<r>/record-set.cbor`); `RecordSet::content_hash` is available for content
  addressing when the r2/iroh blob plane lands (the MERGE-2 `verify_record_set` object-fetch note).
