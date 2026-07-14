# Swarm P2 — Lane B2+B3 ledger (observability + performance)

Lane **B2+B3** of the **Swarm P2 WAN Program**, Wave 2. Branch `swarm/b2b3`, base `4e821cd`
(trunk `integrations/swarm-p2` Merge-1 HEAD), worktree `/home/j/experiments/daemon-worktree/p2-b2`.
This lane lands two of the three Wave-2 B items:

- **B3 — the perf follow-on:** remove the per-op host-readback tax in `BurnBackend` (P1 M2 measured
  a 2.3–2.6× tokens/s tax vs a straight-burn reference at 160M on wgpu) by keeping native op results
  **device-resident** and materializing host copies only at genuine host boundaries.
- **B2 — observability wiring:** wire `daemon-swarm-observe` (MessageLog + replay oracle + desync
  tally + run-health) into the live runtime — an `--observe <dir>` flag on `swarm-local` and a
  `swarm-replay` verification entry point (the gate-ceremony instrumentation).
- **RUN-10 carried item:** the additive `Manifest.max_round_interval_ms` staleness ceiling + its
  assess-time screen + the carried `demo_module_ineligible_on_slow_coordinator` test + guest rebuild.

Program ledger: `swarm-p2-ledger.md`. P1 numeric baseline: `swarm-p1-throughput.md`. ABI (frozen —
host-side work only, no new ops): `swarm-tensor-abi-spec.md`. RUN-10 carry-in: `swarm-ledger-p2-b1.md`.

## Owned areas (disjoint by construction)

`crates/coprocessor/daemon-train` **src** (`OpBackend`/`BurnBackend`/engine internals) + its
`tests`/benches; `crates/swarm/daemon-swarm-observe` + its wiring into `daemon-swarm-run`'s
`harness`/`live_harness`/`bins`; `crates/contracts/daemon-train-sdk` (the `Manifest` field);
`guests/` (manifest regen after the SDK `Manifest` change). NOT touched: the `daemon-train-worker`
**bin** (transport/worker glue is A3's this wave), `daemon-swarm-net`/`daemon-swarm-node` (A1/A3),
the frozen root files (`Cargo.toml`/`deny.toml`/`flake.nix`), daemon-cloud (C).

`tabi@1` is **FROZEN** (66 ops). This lane adds **no ABI ops** — B3 is an internal representation
change behind the frozen `OpBackend` trait; RUN-10 is a config/SDK `Manifest` field, not a wire/ABI
change.

---

## B3 — lazy device-resident `OpBackend` (the perf follow-on)

### Root cause (the P1 tax, re-confirmed by A/B)

`BurnBackend` held each tensor as `{ t: Tensor<B,1>, host: Vec<f32> }` and `insert_result` ran
`to_vec_f32` (`to_data`) after **every** native op — a device→host readback + queue flush per op on
wgpu that serialized the GPU pipeline (P1 `swarm-p1-throughput.md §"Honest overhead accounting"`).
The straight-burn reference keeps tensors on-device across a step, so it pipelines. The tax grows
with ops/step (CPU 1.65×→2.05×; GPU 2.33× at 160M).

### Design — lazy host cache via `std::cell::OnceCell` (no trait change)

The frozen trait's `view(&self, id) -> &[f32]` returns a **borrowed** slice, which blocks a plain
`&mut`-materialize-and-cache. Resolved **without extending the trait** using interior mutability:

- `Slot.host: OnceCell<Vec<f32>>` (was `Vec<f32>`).
- `insert_result(t)` stores `Slot::lazy(t)` — **no `to_data`**; the value stays device-resident.
- `view`/`host(&self, id)` calls `slot.host.get_or_init(|| to_vec_f32(&slot.t))` — the host copy is
  materialized **once, on first read**, and cached, returned as `&[f32]` with the `&self` lifetime.
- Leaves (`create`/`zeros`/`write`) seed the cache **eagerly** (`Slot::eager` — the caller already
  holds the bytes: params, det results, requantized masters, checkpoint restores).
- `adamw_step` now stores `m`/`v`/`master` **lazy** (was 3 readbacks/param/inner-step): `master` is
  materialized on demand by the caller's storage sync (`op_adamw_step` reads `view(master)` once);
  `m`/`v` only materialize at a checkpoint boundary.

**Host boundaries that materialize (the residency inventory — a Merge-2 seam):** the det lane (every
`det_*` op + compression native runs `det_core` on host fp32), scalar/metric readouts (loss),
`canonical_state_bytes` (the consensus digest), `checkpoint_bytes`, `upd_push_tensor` (make_update
staging), `grad@1` fold, and any `to_data` for `MetaReport`. Everything else — the entire forward
matmul/attention/norm chain and burn's internal backward — stays on-device.

`OnceCell<Vec<f32>>` is `Send` (the trait's only supertrait bound; not `Sync`, not required). The
frozen `OpBackend`/`TrainerBackend` traits are **unchanged**; no additive method was needed.

### Correctness — det/parity bar held byte-identically

The materialized host bytes are `to_vec_f32(&t)` on the same immutable burn tensor whether read
eagerly or lazily, so every downstream det/digest/parity value is identical. Verified:

- `cargo test -p daemon-train --features burn-ndarray`: `burn_backend_parity` (18), the G2
  `cross_backend::cross_backend_det_digest_{demo,diloco,sparse_loco}` (byte-identical CPU-vs-burn det
  digests), `wasm_backend_determinism` (12), `checkpoint_save_load_continue_matches_uninterrupted`,
  `worker_protocol` (4) — all green.
- `.#vulkan -p daemon-train --features wgpu`: `burn_wgpu_parity` **18/18** (incl. `det_lane_bit_exact`,
  `compression_natives_bit_exact`, `abi_adamw_step_matches_burn`, every forward/backward parity),
  `wgpu_lifecycle` **3/3**.
- **160M wgpu reference-parity rerun** (`reference_parity_wgpu --ignored`): per-step loss
  **byte-identical** (|Δ| = 0.000e0 for all 4 steps), final-weight max Δ = **4.768e-7** (Optimizer
  class rtol 2e-4/atol 2e-5) — the det lane stays host-side fp32 and the digest is unchanged.

### Throughput — the headline (rigorous A/B, identical methodology, same session)

160M preset (`llama_160m`, d768/L12/seq1024, 151,862,784 params), wgpu on RADV (Strix Halo), b=1 over
real TinyStories, **3 warmup + 10 measured** steps (mean ± sd). The eager row swaps in the base
commit's `burn_backend.rs`; the reference path is unchanged, and its near-identical number across the
two runs (735 vs 753 tok/s) confirms the A/B is machine-state-fair.

| Backend | Config | tabi tok/s | tabi step | reference tok/s | tabi/reference |
|---|---|---:|---:|---:|---:|
| wgpu RADV | 160M (eager, base) | 253.6 | 4.034 s ± 1.014 s | 735.4 | **2.90×** |
| **wgpu RADV** | **160M (lazy, B3)** | **383.9** | **2.665 s ± 0.080 s** | 753.4 | **1.96×** |

**B3 result:** +51% tabi tokens/s (253.6 → 383.9), the overhead **factor's excess-over-1.0 roughly
halved** (1.90× → 0.96×), and tabi per-step variance cut ~13× (±1.01 s → ±0.08 s — the eager per-op
readbacks were the variance source: queue-flush stalls). Under the P1 methodology (1 warmup + 4
measured) the lazy backend reads 2.26× vs P1's recorded 2.33×; the low-variance A/B above is the
honest measure.

### What remains of the host-copy tax (the honest residual, 1.96×) — and why

The forward/backward is now pipelined like the reference. The residual ~0.96× excess is **not** the
per-op activation readback (removed); it is the **host-side fp32 residency contract itself** (ABI
§5.9), which both paths pay but tabi pays more granularly:

- **Param grad host-fold** (`op_backward`): each param's gradient is read back (`grad_of`) and folded
  into the host `grad` accumulator so it survives micro-batch splits — a per-param round-trip. The
  reference reads grads too, but folds fewer.
- **Master/storage sync** (`op_adamw_step`/`det_axpy_param`): the fp32 master is host-authoritative
  and mirrored to a separate `storage` leaf (re-uploaded) so the next pass differentiates through it.
- **The det + compression boundary** (`make_update`/`ingest`): `dct2`/`topk_chunk`/`absmax_pack`/the
  `det_*` aggregate all materialize host fp32 by design (the consensus digest must be
  backend-independent and bit-exact) — unavoidable at the ingest doorway, and absent from the
  reference (which never compresses).

Removing these would change the frozen §5.9 residency contract (host-authoritative fp32 masters +
the backend-independent det digest) — out of scope and undesirable (they are the det-lane exactness
guarantee). **Verdict:** B3 removed the tax that was *not* load-bearing (per-op activation readback);
the residual 1.96× is the load-bearing det/residency cost, honestly attributed.

### Bench discipline

`swarm-p2-throughput.md` records the before/after table (ndarray reduced/medium + wgpu 160M) with the
deterministic timed-loop method (warmup + stated variance), extending the `swarm-p1-throughput.md`
style. The throughput test gained env knobs `M2_WGPU_WARMUP`/`M2_WGPU_MEASURED` (defaults unchanged,
so the P1 gate is byte-identical) for the low-variance evidence run.

---

## B2 — `daemon-swarm-observe` wired into the runtime

The observe crate was already a complete library (MessageLog, replay oracle, desync tally,
run-health). B2 **wires it into the live runtime** and adds the recorded-run replay path.

### Additive observe surface (freeze at Merge 2)

- `daemon_swarm_observe::RunCapture { initial: CoordinatorState, inputs: Vec<Input> }` (new
  `capture` module) — the coordinator's reproducible `tick` driver trace (messages **and** clocks;
  clocks never cross the wire, §14, so the node-visible MessageLog alone cannot replay a
  clock-driven coordinator). `write_to`/`read_from` frame it as `magic + canonical-CBOR`.
- `replay_from_state(initial, inputs)` — the replay oracle from a given genesis state (the existing
  `replay(env, params, …)` now delegates to it after resolving the envelope). `replay_capture(capture,
  &log)` — re-runs `tick` over the capture's driving trace and compares against the **independent**
  wire `MessageLog`'s `RoundRecord`s (the oracle), so a green replay proves per-round consensus
  (committed set + drops = the digest) is byte-reproducible. `logged_round_records(&log)` helper.
- `daemon_swarm_run::harness`: `SwarmRun.message_log` (additive field — every verified `SignedMessage`
  captured in arrival order via `spawn_message_log`, generic over `ControlPlane` so loopback + live
  reuse it); `SwarmRun::run_capture()` / `SwarmRun::write_observe(dir)` (writes `<run>.dsmlog` +
  `<run>.dsmcap`); `verify_observe_dir(dir) -> ObserveVerify` (the `swarm-replay` library entry, also
  projects `RunHealth`). `CoordinatorReplay::{initial_state, inputs}` accessors (additive).
- **CLI:** `swarm-local --observe <dir>` writes the artifacts after a run; **`swarm-replay <dir>`**
  (a new bin, `required-features = ["harness"]`, transport-free) re-derives + verifies + prints
  per-round health, exit-non-zero on divergence.

### Tests (record + replay green)

- `daemon-swarm-observe/tests/observe.rs::run_capture_replays_recorded_run` — the RunCapture
  round-trips byte-identically and `replay_capture` re-derives all 3 records + a byte-identical final
  state hash.
- `daemon-swarm-e2e/tests/swarm_e2e.rs::observe_record_and_replay_green` — loopback: record a
  20-round run, `verify_observe_dir` re-derives 20/20, digest tally shows no desync outliers, health
  projects 20 finalized rounds.
- `daemon-swarm-e2e/tests/live_transport.rs::live_observe_record_and_replay_green` — the requested
  **live_transport-suite variant**: an 8-round run over the real iroh mesh, recorded + replayed
  green (8/8 re-derived). live_transport now **7/7** (6 prior + this).

The MessageLog captured off the harness plane carries the coordinator's `RoundOpen`/`RoundRecord` +
peers' `Commitment`/`Heartbeat`/`Join`/`Straggle` (peers report post-ingest digests via engine
events, not wire `Digest` messages, so the log's digest-tally is exercised on the desync-injection
path; the recorded-run digest-equality assertion rides the `RoundRecord` replay oracle).

---

## RUN-10 — `Manifest.max_round_interval_ms` (carried from B1/Merge-1 Decision 3)

Additive, optional, config/SDK-level (**not** the SwarmApi wire, **not** `tabi`):

- `daemon_train_sdk::api::Manifest.max_round_interval_ms: Option<u64>` (`#[serde(default)]`,
  `Manifest::new` → `None`) — the **staleness ceiling**, the mirror of the existing
  `min_round_interval_ms` floor. `daemon_train::runtime::Manifest` gains the matching `#[serde(default)]`
  field so an older module's `da_manifest` CBOR (no key) still decodes to "any cadence".
- `DemoProfile::manifest` declares `Some(2000)` (a real-time per-step demo, §5.3.3, is stale if a
  round takes > 2 s); `sparse_loco` declares `None` (tolerates any cadence).
- `daemon_swarm_run::assess::screen_round_cadence(max, coordinator_interval_ms) -> RoundCadence`
  (`Eligible` / `TooSlow`) — the assess-time soft screen (§6.5): a module is ineligible when the
  coordinator cadences slower than its ceiling.
- Tests: `assess::tests::demo_module_ineligible_on_slow_coordinator` (the carried B1 item —
  ineligible at 5 s, eligible at ≤ 2 s, `None` ⇒ any) and
  `profiles::run10_tests::demo_manifest_declares_staleness_tolerance` (pins the demo's ceiling +
  floor at the SDK source).

The SDK `Manifest` change recompiled the guest `.wasm`; `cargo run -p xtask -- build-guests` was
re-run and **`guests/guests.blake3` re-committed** (guard is warn-and-rebuild per the Merge-1
adjudication — the canonical trunk manifest is kept fresh).

---

## Seams this lane exports (freeze at Merge 2)

1. **The lazy `OpBackend` internal contract** — the host-boundary inventory above (the set of call
   sites that materialize host fp32). The `OpBackend`/`TrainerBackend` traits are unchanged; the
   residency contract (host-authoritative fp32 masters + backend-independent det digest, ABI §5.9)
   is unchanged and remains the correctness bar.
2. **The observe wiring surface** — `--observe <dir>` (writes `<run>.dsmlog` + `<run>.dsmcap`),
   `swarm-replay <dir>` (verifies), `SwarmRun::{message_log, run_capture, write_observe}`,
   `verify_observe_dir`, and `daemon_swarm_observe::{RunCapture, replay_from_state, replay_capture,
   logged_round_records}`.
3. **`Manifest.max_round_interval_ms`** (SDK + runtime) + `assess::screen_round_cadence`.

## Deviations / notes for Merge 2

- **B3's win is smaller than the naive 2.3× → ~1.0×** the P1 doc's framing might suggest, because the
  per-op activation readback was **not** the whole tax — the host-side fp32 residency (grad-fold +
  master/storage sync + det/compression boundary) is load-bearing and shared. The honest post-change
  ratio is **1.96× at 160M** (excess halved), documented above and in `swarm-p2-throughput.md`.
- **`#[allow(clippy::disallowed_methods)]`** is scoped to `write_observe`/`verify_observe_dir` (plain
  local-fs on an **operator-supplied** gate directory, not attacker-influenced; `harness`-gated
  dev/gate tooling). The clippy fs ban targets attacker-influenced paths via `ContainedRoot`; the
  lint layering keeps network separately locked (disallowed-TYPES), so this narrow fs escape is safe
  — the same exception the e2e test files already take. If Merge-2 prefers, route through
  `ContainedRoot` (adds a `daemon_core` dep to `daemon-swarm-run`).
- **No frozen-file edits.** `daemon-swarm-run/Cargo.toml` gained a second `[[bin]]` (`swarm-replay`,
  `required-features = ["harness"]`) and the observe crate declared `capture` — both lane-owned.
- **No wire/ABI change**, so no `WireVersion` bump (stays v42).
- Known flake untouched: `daemon-conformance` detached-delegation trio (no swarm lane touches it).
