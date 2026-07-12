# Swarm-training MVP — lane P2 ledger (consensus / coordinator)

Wave-2 working record for lane **P2** (`swarm/p2`). Scope is the deterministic assignment math (in
`crates/contracts/daemon-swarm-proto`, additively) and the purified coordinator `tick` state machine
+ admission logic (in `crates/swarm/daemon-swarm-coordinator`). Read the program ledger
([`swarm-mvp-ledger.md`](swarm-mvp-ledger.md)) first — especially **"Merge 1 — frozen interfaces"**:
the Wave-1 proto API is FROZEN and extended **additively** only. Lane P1's ledger
([`swarm-ledger-p1.md`](swarm-ledger-p1.md)) documents the frozen proto surface this lane builds on.

## Base + branch

- **Repo:** `daemon-node` (Rust backend submodule; standalone checkout).
- **Base commit:** `c1432fa` (`mirror(merge-1): freeze cross-lane interfaces`) — the Merge-1 tip on
  `integrations/swarm`, which integrated all three Wave-1 lanes (P1/R1/E1).
- **Branch:** `swarm/p2`, forked from `c1432fa`. Integrates back into `integrations/swarm` at Merge 2.
- **Worktree:** `/home/j/experiments/daemon-worktree/swarm-proto` (isolated; never touches the main
  read-only checkout at `/home/j/experiments/daemon`).

## Scope this wave (lane P2, Wave 2)

Two crates this lane owns outright (including their `Cargo.toml` dep sets): the **new**
`assignment` module inside `daemon-swarm-proto` (purity-first, so the future replay oracle can
consume it) and the whole of `daemon-swarm-coordinator`, kept a **library** this wave (pure `tick` +
types). The runnable local coordinator binary/loop is Wave 3 (lane R owns `bins/`).

| Area | Spec / TDD grounding | Deliverable |
|---|---|---|
| Seeded shuffle + committees | §6.3, A.2/A.3; TDD PROTO-4, golden shuffle vectors | `assignment` module (in proto): `Lcg`, `deterministic_shuffle`, `select_committee`, `witness_quorum` |
| Throughput-class-weighted batch windows | §6.3; TDD PROTO-3/8 | `assign_batches` (roster+classes+window → per-peer `BatchWindow`), `global_batch_at` ramp, `advance_cursor` |
| Purified `tick` state machine | §6.2, §6.4, §11.2; TDD PROTO-1/2/3/14 | `tick(state, input) -> (state', outputs)`: I/O-free, clock-as-input, canonical-CBOR-serializable |
| Round protocol + commit rule | §6.4 (I1–I6); TDD PROTO-5/6/7 | pure commit rule over signed evidence; stall ladder + K-record-absence drops |
| Admission | §6.5; TDD PROTO-12/13 | `admit()` + typed reject reasons (version, envelope hash, capability subset, roster, phase) |
| Termination / epoch / pause | §6.1/§6.2/§11.1; TDD PROTO-9/10/14/15/17 | ramp + `stop` → Cooldown → Finished; epoch boundaries; authorized pause/resume |
| Replay foundation | §6.4 I1; TDD PROTO-20 | canonical-CBOR round-trip of `CoordinatorState`; byte-identical `(state', outputs)` under replay |

## Design decisions (rationale for choices the spec left to the lane)

1. **Assignment lives in `daemon-swarm-proto`** (not the coordinator). Spec §6.2/§6.4 make replay a
   pure function of `(checkpoint, records, payloads)`, and §6.3's assignment is what a replay oracle
   re-derives per round. Placing it in the wasm-clean proto crate keeps it consumable by the oracle,
   the coordinator, and every peer without a coordinator dependency. The coordinator **re-exports**
   the pieces it uses. (Spec §6.3 says the boundaries are "`daemon-swarm-proto` constants" — the
   class-weight ladder and `WITNESS_TARGET_DEFAULT` land there too.)

2. **`tick` lives in `daemon-swarm-coordinator`** (per the lane brief), kept **pure** and wasm-clean:
   `tick(state, input) -> (state, Vec<Output>)`, no I/O, no clock read — **time enters as an
   `Input::Clock(unix_s)`**. This is the Risc0/zkVM-substrate purity note (§11.2, §18 q.14) and the
   replay-oracle foundation (I1/PROTO-20). Spec §6.2 phrases the pure function as living "in
   `daemon-swarm-proto`"; the lane brief places it in the coordinator crate, which is where it is —
   the purity property is preserved either way (deviation noted below).

3. **`tick` never signs.** A pure, secret-free transition cannot hold a `SigningKey` in
   canonical-CBOR state. So the coordinator emits its own messages as **unsigned** `Output::Publish(
   SwarmMessage)` values (`RoundOpen`/`RoundRecord`); the Wave-3 harness signs them with the
   coordinator identity before broadcast (ed25519 is deterministic, so replay stays byte-identical).
   Inbound messages arrive already-signed (`Input::Message(SignedMessage)`) and `tick` verifies them
   purely (ed25519 verify is I/O-free). This is what keeps the commit rule "a pure function of signed
   messages" (I6) while `tick` stays key-free.

4. **Commit-rule coverage uses the attestation `inline` set** (the small-roster transport path,
   §6.4). A bare merkle root cannot answer "is `(peer,hash)` in this witness's set?" without a proof;
   the MVP small rosters ride the inline list, so `tick` reads `Attestation.inline` for quorum
   coverage. The signed/consensus field is still the root; the root-only path (coordinator holds
   membership proofs) is a Wave-3 extension. `StorageReceipt` evidence needs no inline (it carries
   the `(peer,hash,size)` tuples directly).

5. **Coordinator crate goes dependency-lean + wasm-clean.** The Wave-0 scaffold declared
   `axum`/`tokio`/`thiserror`; a pure library `tick` needs none of them, and dropping them keeps the
   crate on the `wasm32-unknown-unknown` substrate path (COORD-3). This wave the crate depends on
   `daemon-swarm-proto` + `serde` only, with a **hand-rolled** `CoordinatorError` (matching the
   lean-contract convention proto uses). Wave 3 (lane R) re-adds the server deps when it wires the
   runnable coordinator binary. This edits only the crate's own `Cargo.toml` (lane-owned) — the root
   `Cargo.toml`/`deny.toml`/`flake.nix` are untouched (FROZEN).

## Additive proto extensions (freeze at Merge 2)

**None to existing types/CDDL.** The `assignment` module is a **new** module (new functions + a new
`Committee`/`BatchAssignment` return shape); it adds no field to any frozen message, envelope, or
CDDL rule, so the Merge-1 wire contract is byte-for-byte unchanged. Everything the coordinator needs
from the frozen `Join`/round-message set it reads from existing fields (see the admission note).

## Seams this lane exports (freeze at Merge 2)

- **Assignment (proto):** `Lcg`, `seeded_lcg(seed, salt)`, `deterministic_shuffle(&mut [T], &mut Lcg)`,
  `witness_quorum(n) -> u32`, `select_committee(roster, seed, witness_target) -> Committee`,
  `class_weight(ThroughputClass) -> u64`, `global_batch_at(GlobalBatch, round) -> u64`,
  `advance_cursor`, `assign_batches(roster, seed, window, overlap_bps) -> Vec<(PeerId, BatchWindow)>`.
- **Coordinator:** `CoordinatorState` (+ `Phase`, `Member`, `ClientState`, `RoundView`), `RunConfig`
  (+ `CoordinatorParams::from_envelope`), `Input` (+ `ControlRequest`/`ControlAction`), `Output`
  (+ `Rejection`, `AdmissionReject`, `Notice`), `tick(state, input) -> (CoordinatorState, Vec<Output>)`,
  `admit(...)`, `CoordinatorError`.

## Planned commit slices (each green per the gates; TDD tight test+impl slices)

1. `mirror(P2): ledger` — this file.
2. `feat(swarm-proto): deterministic assignment — LCG, shuffle, committees, batch windows (green)`.
3. `feat(swarm-coordinator): pure tick state machine + round protocol + commit rule (green)`.
4. `feat(swarm-coordinator): admission + pause/resume authorization + rejoin (green)`.
5. `mirror(P2): ledger — Merge-2 seams + results` (final, after the full-workspace gates).

Slices may merge if the gates dictate; every commit passes `cargo fmt --check`,
`cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`,
`cargo build --target wasm32-unknown-unknown -p daemon-swarm-proto`, and `typos docs/specs`.

## Deviations from spec/TDD names (recorded)

- **Swap-or-not shuffle not ported byte-for-byte.** The TDD lists Psyche's `swap_or_not.rs` /
  `deterministic_shuffle.rs` / `lcg.rs` as DIRECT ports with GOLDEN vectors, but the upstream sources
  are not vendored in this repo. This lane implements a documented 64-bit LCG (Knuth MMIX constants)
  + Fisher–Yates `deterministic_shuffle`, and pins **its own** golden vectors from the daemon seed
  `0xDAE07E57` (a daemon-specific seed, not a Psyche vector). Determinism/replayability — the
  load-bearing property — is what is tested; exact Psyche byte-parity is out of reach without the
  upstream and is not required by the daemon golden seed.
- **`tick` in the coordinator crate, not proto** (per the lane brief) — see decision 2.

## Notes for Merge 2 integration (what to watch)

- The coordinator `tick`/`CoordinatorState`/`Input`/`Output` types are the Wave-3 wiring seam (lane R
  drives the harness loop; app mirror consumes projections). Freeze them at Merge 2.
- `tick` emits **unsigned** coordinator messages; the harness must sign+publish (decision 3).
- Admission reads only frozen `Join` fields; **envelope-hash admission is structural** (reject reason
  + optional `asserted_hash` arg on `admit`) but not enforced from the frozen `Join` (which carries
  no hash). Wire it when `Join` gains an (additive) `envelope_hash` field or an assessment token.
- Coordinator-only knobs absent from the frozen envelope — `witness_target` (default 4, §6.3),
  `overlap_bps` (0–10%, §6.3), `k_absences` (K record-absences drop, §6.4 daemon Delta),
  `seq_len` (tokens/sequence, for `stop.tokens`), `verification_percent` (default 0, §12) — are
  carried in `CoordinatorParams`, supplied at run creation (Wave-3 authoring), not read from
  `[experiment.config]` at runtime (seam rule §4.3 preserved).
