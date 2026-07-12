# Swarm-training MVP — lane E3 ledger (engine / tensor-ABI / guests, Wave 3)

Wave-3 (final wave) coordination record for lane **E** (`swarm/e3`). Companion to the program
ledger (`swarm-mvp-ledger.md`) and the Wave-1/2 lane records (`swarm-ledger-e1.md`,
`swarm-ledger-e2.md`). Read the program ledger's FROZEN-file + file-ownership rules and the
"Merge 1"/"Merge 2" frozen-interface sections first — they bind this lane unchanged. Wave-3 extends
every frozen surface **additively only**: `tabi@1` stays at 66 ops, the phase table is unchanged,
`det-core` and the SDK grow nothing this wave.

## Base + branch

- **Branch:** `swarm/e3`, forked at `39c0ebd` (`mirror(merge-2): P0 milestone — freeze Wave-2
  interfaces`) on `integrations/swarm` — Merge 2, all Wave-2 lanes (P2/R2/E2) integrated.
- **Merge target:** `integrations/swarm` (Merge 3). Disjoint file set → conflict-free with the other
  Wave-3 lanes (R3 owns the local runner + drills; P3 owns coordinator/observe).

## Scope (this lane owns; edits confined here)

| Path | Wave-3 role |
|---|---|
| `crates/coprocessor/daemon-train/` | the **E↔R wiring**: `WasmBackend` (a `daemon_swarm_run::TrainerBackend` over the wasm host runtime) + the `daemon-train-worker` binary (frozen worker protocol over stdio) |
| `crates/contracts/det-core/` | unchanged this wave (parity upkeep only) |
| `crates/contracts/daemon-train-sdk/` | unchanged this wave (parity upkeep only) |
| `guests/` | `guests/README.md` authoring guide; modules unchanged |
| `xtask build-guests` | unchanged |

FROZEN (never touched): root `Cargo.toml`, `deny.toml`, `flake.nix`, and every other lane's
directories. **No new third-party dependency** is introduced — the crates added to
`daemon-train`'s manifest (`daemon-swarm-run`, `daemon-provision`, `tokio`, plus dev-deps
`daemon-train-client`, `daemon-swarm-net`) are all already pinned in the frozen root
`[workspace.dependencies]` as path/workspace deps, so `daemon-train`'s `Cargo.toml` gains only
`{ workspace = true }` lines (a lane-owned manifest edit, no root/deny/flake change, no
`cargo deny` re-run). No dependency cycle: `daemon-swarm-run`/`-net`/`daemon-train-client` do **not**
depend on `daemon-train`/`-sdk` (verified at Merge 2), so `daemon-train → daemon-swarm-run` is acyclic.

## The E↔R wiring (what makes the round loop drive real WASM training)

`daemon_swarm_run::TrainerBackend` is the engine-agnostic seam the peer `RoundEngine` drives (opaque
bytes + plain structs across it; no wasmtime/burn types leak). Wave 1/2 exercised it with
`StubBackend`; Wave 3 lands the real thing.

### `WasmBackend` (`daemon-train/src/wasm_backend.rs`) — FROZEN at Merge 3

The constructor takes the module `.wasm` bytes + an `EngineConfig`; `build(config)` instantiates via
the `InstancePre` re-instantiation path and runs the `da_abi` gate → `da_manifest` (parsed cadence)
→ `da_build`. The lifecycle maps 1:1 onto the host `Instance`:

| `TrainerBackend` | host `Instance` | notes |
|---|---|---|
| `build(cfg)` | `instantiate` + `da_abi` gate + `manifest` + `da_build` | constructor holds the `.wasm`; `build` holds the config for churn re-build |
| `train_step(batch, ctx)` | `register_batch` + `da_step` | `batch.tokens.len()/seq_len` sequences, phase/fuel/epoch/op budgets enforced host-side |
| `inner_update(step)` | `da_inner_update` | |
| `make_update(round)` | `da_make_update` → `update_bytes(handle)` | seals the `upd_*` container to canonical CBOR (the opaque payload the swarm moves; never parsed) |
| `ingest(round, staged)` | `ingest_payloads(round, [bytes…])` | stages the record-ordered payloads through the `upd_*` ABI (one container per staged payload, in caller order), then `da_ingest_updates` |
| `checkpoint_save/load` | `checkpoint_bytes` / `restore_checkpoint` | blake3-integrity-tagged full state dict |

`assess(meta)` returns a footprint/eligibility estimate cached from `da_build` (params fp32 masters +
grads + a payload estimate); the worker binary's `AssessRun` uses the richer host **meta mode**
(`Instance::meta`) directly instead.

**Digest** = the host's canonical state digest over `canonical_state_bytes()` (params fp32 masters,
then `class = 1` **replicated** native persistents, then `class = 1` replicated det persistents, in
registration order), fed through `daemon_swarm_proto::digest::digest_state` (seed-keyed xxh3-128,
seed derived from the round, `block_size = 64`, full sampling so the whole canonical state is
covered — a full digest, not a sample, since the MVP wants exact cross-peer identity). The proto
digest module is frozen; this lane consumes it and does **not** re-derive a second hasher. Local
(`class = 0`) persistents (AdamW moments, sparse_loco error-feedback, demo momentum) are **not**
digested — peers rebuild them (ABI §5.1), matching `daemon_train_sdk::Persistent::local`.

### Checkpoint format (documented seam) — FROZEN at Merge 3

`checkpoint_save()` bytes are `blake3(body) ++ body`, where `body` is canonical CBOR of
`{ params: [[f32]], round_base: [[f32]], persistents: [[f32]], det_persistents: [[f32]] }` — the
**full** worker state dict (every param master, its round base, and **all** native + det persistents
regardless of class), in registration order. `checkpoint_load` recomputes + verifies the blake3
prefix, then restores masters/storage/round-base and every persistent bit-exactly.

Deviation from the query's minimal "masters + replicated det persistents + step counters" list
(recorded): the checkpoint stores the **full** state (including `class = 0` local persistents), not
just the replicated consensus subset. Rationale — the MVP's own acceptance test is
*save→load→continue matches the uninterrupted run bit-for-bit*, and the continuation's very next
inner steps read the AdamW moments (local persistents); dropping them would silently diverge the
continued digest. The replicated subset a cross-peer **resync/rejoin** needs is a subset of what we
store, so nothing is lost; storing the full dict is strictly safer. There is no separate step
counter in the state dict — a round-boundary checkpoint has `round_base == master`, and the round id
is carried by `CheckpointManifest` (daemon-swarm-run), not the opaque bytes.

### Preemption-as-churn (§10.5, T3) — `WasmBackend::pause` / `resume`

`pause()` checkpoints the current state, then drops the wasm instance (releasing all wasm memory /
GPU allocations, keeping only the CPU-side checkpoint). `resume()` re-instantiates from the same
`InstancePre`, re-runs `da_build` (fresh registration, deterministic under T3), and
`restore_checkpoint`s the saved state — bit-identical to the pre-pause state. The worker maps
`Throttle{paused:true/false}` onto these. Because a checkpoint is a round-boundary snapshot, an
in-flight round interrupted by the epoch watchdog (a `BudgetEpoch` trap on the aborted guest call)
is discarded and redone from the last committed round base — the churn is digest-neutral.

## The `daemon-train-worker` binary — FROZEN at Merge 3

**Binary name:** `daemon-train-worker` (a new `[[bin]]` in `daemon-train`; the Wave-0
`daemon-train` version-line bin is left untouched). **Invocation contract** (what a supervisor
spawns): the length-framed CBOR stdio cut (`daemon_provision::CutChannel`, `Framing::Length`) exactly
as the fake worker; the experiment `.wasm` module is located via the `DAEMON_TRAIN_MODULE` env var
(an absolute path). It speaks the frozen `daemon_swarm_run::protocol` `Command`/`Event` set:

- **startup** → `Event::Ready { capabilities }` (tabi@1, all 66 ops from the host vocabulary,
  `payload_stores: []`).
- `Probe` → `Event::Probed(Hardware)`: a real host capability report — `abi_version = 1`, the 66-op
  `tabi@1` vocabulary, GPU absent (`gpus = 0`, `backend_lanes = ["cpu"]`, CPU-only class `c1`).
- `AssessRun { envelope }` → the peer-side re-validation (spec §6.5): a **static import scan** of the
  module bytes vs the host vocabulary (a module importing an op outside `tabi@1` is ineligible),
  then a host **meta-mode** pass (`Instance::meta`) → `Event::Assessed(Eligibility)`. For the MVP the
  envelope bytes are the experiment `[experiment.config]` CBOR directly (real `FrozenEnvelope`
  parsing + artifact resolution is the Merge-3 wiring point noted below); the config is cached for
  the subsequent `JoinRun`.
- `JoinRun { … }` → constructs a `WasmBackend`, emits `Event::RunPhase{phase:"train"}` (the
  supervisor's `join` resolves here), then **self-drives one round** (train × `steps_per_round` →
  `make_update` → `ingest` of its own payload), emitting `Event::Metric{loss}` +
  `Event::RoundOutcome{round, committed:1, ingested:1, digest}`.
- `Throttle{paused}` → `WasmBackend::pause`/`resume` (preemption-as-churn).
- `Leave` / `Shutdown` / `Ping` → as the protocol requires.

A trapping module (typed `daemon_train::Trap`) becomes `Event::Error{ class: Module, detail }` — the
worker is never harmed (ABI §3.6 / §13).

## Additive host surface (new public API on `daemon-train`, all additive)

- `Instance::update_bytes(container) -> Vec<u8>` — seal a `da_make_update` container to canonical CBOR.
- `Instance::ingest_payloads(round, &[Vec<u8>])` — stage record-ordered payloads + `da_ingest_updates`.
- `Instance::canonical_state_bytes() -> Vec<u8>` — the digested state (params + replicated persistents).
- `Instance::checkpoint_bytes()` / `restore_checkpoint(&[u8])` — blake3-tagged full-state (de)serialize.
- `Instance::imports_charged() -> usize` — host-ops charged so far (the HOST-15 manifest-purity probe).
- `OpBackend: Send` (supertrait added) — so `WasmBackend` is `Send` (the `TrainerBackend` bound). The
  only impl (`CpuBackend`) is already `Send`; no behavior change.

Nothing existing changed signature or semantics; the phase table, `tabi@1` (66), `det-core`, and the
SDK are untouched (`abi_surface` stays green at 66).

## Determinism evidence (the MVP's core claim)

`daemon-train/tests/wasm_backend_determinism.rs`:

- **Cross-PEER bit-identity** (the guarantee): two `WasmBackend`s over the same module + config +
  batches + staged payloads produce **bit-identical** digests after N rounds of
  step/inner_update/make_update/ingest — one case per profile (`sparse_loco`, `diloco`, `demo`).
- **Checkpoint continuity**: `save → load → continue` reaches the same digest as the uninterrupted
  run (and `pause → resume` — preemption-as-churn — is digest-neutral).
- **Sim ↔ host parity**: a nice-to-have cross-check, **not** a hard guarantee (recorded rationale
  below). The det-lane ingest (the agree-path) shares `det-core` between sim and host, so it is
  bit-identical; the **native lane** (make_update's compression math + AdamW) runs a different tape
  in the sim (`daemon-train-sdk::sim`) than the host `CpuBackend`, so the peer-distinct
  `make_update` contribution — and hence a full end-to-end sim-vs-host digest — need **not** be
  bit-equal. The cross-peer guarantee (WasmBackend vs WasmBackend, one implementation) is what the
  MVP requires and what is asserted; sim-vs-host equality is asserted only where the ABI contract
  requires it (the det-lane fold), documented in the test.

## Module hygiene (SDK/guests)

- **HOST-15 manifest purity**: `da_manifest` charges **zero** host imports (a fresh instance's
  `imports_charged()` is 0 after `manifest()`) for tiny-llama — extended from the Wave-1 pattern. A
  manifest that called a host import would trap `PhaseViolation` (no phase is entered for `da_manifest`).
- **Guest `.wasm` size**: recorded by `guest_wasm_sizes_are_sane` (release build, well under a few
  hundred KB — see the lane report for the measured bytes).
- `guests/README.md` documents authoring an experiment (the SDK surface, the `experiment!` macro, the
  three profiles + `TinyLlamaCfg`, and building via `cargo run -p xtask -- build-guests`).

## Sim/native parity upkeep

Additive only: no det-core or SDK signature/semantic change was needed. `abi_surface` stays at 66
ops (no profile required a new op). No new sync-point drift.

## Planned slices (commit order; each lane-scoped green)

1. `mirror(E3): ledger` — this file.
2. `feat(train): WasmBackend over the wasm host runtime + determinism (green)` — the additive host
   methods, `WasmBackend`, and `wasm_backend_determinism.rs`.
3. `feat(train): daemon-train-worker binary speaking the frozen worker protocol (green)` — the bin +
   `worker_protocol.rs` integration test (driven through `TrainSupervisor`).
4. `feat(train): module hygiene — manifest purity + guest sizes + authoring guide (green)`.

## Merge-3 watch list (what integration must verify)

- **Real-backend wiring is now live**: point R3's local runner's `RoundEngine` at `WasmBackend`
  (in-process) and/or the `daemon-train-worker` binary. The worker binary's round loop is
  **self-driven** for the MVP (no coordinator connection); Merge 3 decides whether the worker
  connects to `JoinRun.coordinator` itself or the node-side runner drives an in-process
  `WasmBackend`. Both are supported by the surface exported here.
- **Envelope / artifact resolution is a Merge-3 seam**: the worker treats the `AssessRun` envelope
  bytes as raw config CBOR and locates the module via `DAEMON_TRAIN_MODULE`. Wire real
  `FrozenEnvelope` parsing (`config_bytes`) + `daemon_swarm_net::ArtifactResolver` (File scheme) at
  Merge 3 so the worker fetches the module from the envelope's artifact.
- **Checkpoint bytes are the full state dict** (superset of the replicated consensus subset). If a
  future wave wants a smaller resync-only checkpoint, it can prune to `class = 1` state — the digest
  contract already only reads that subset.
- **`burn` is still on the default gate** (`daemon-train` declares it); adding `daemon-swarm-run` +
  `tokio` to `daemon-train` widens the crate's default-gate build further. The `OpBackend` seam still
  stands for the eventual lane-split.
- **Cross-peer identity is the frozen guarantee**; sim-vs-host is documented as det-lane-only.
