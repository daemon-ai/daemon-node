# Swarm P3 — Lane G ledger (CUDA engine arm)

Lane **G** of the **Swarm P3 Follow-ons Program** — the `BackendKind::Cuda` engine arm and its
RTX-4090 parity/throughput validation. Worktree `/home/j/experiments/daemon-worktree/p3-g`, branch
`swarm/g`, base `245aef6` (trunk `integrations/swarm-p3` @ Wave-0 — the P3 program ledger on
daemon-node master `64e191a`). This file is the single source of truth for what G landed, the seams
it extends (additively), every dependency/flake note (with rationale), and the RunPod-4090 evidence.
Mirror commit: `mirror(G): ledger`.

Read first, in order: [`swarm-p3-ledger.md`](swarm-p3-ledger.md) (program charter, frozen seams,
fleet endpoints), [`swarm-p2-ledger.md`](swarm-p2-ledger.md) (P2 record), the C2/C3 CUDA findings
([`swarm-ledger-p2-c2.md`](swarm-ledger-p2-c2.md) §D5, [`swarm-ledger-p2-c3.md`](swarm-ledger-p2-c3.md)
§Task-3 — the `.#cuda-train` devshell + nvrtc-12.4 staging), [`swarm-p2-throughput.md`](swarm-p2-throughput.md)
(the B3 lazy-residency method + the 160M wgpu figures this lane compares CUDA against), and
[`swarm-ledger-b3.md`](swarm-ledger-b3.md) (the lazy device-resident host-boundary inventory this arm
must honor).

## Base + branch

- **Repo / worktree:** `daemon-node` @ `/home/j/experiments/daemon-worktree/p3-g`.
- **Base commit:** `245aef6` (`mirror(p3-wave0): program ledger`) on `integrations/swarm-p3` — the
  Wave-0 trunk tip (fast-forwarded my `swarm/g` lane branch onto it so the program charter travels).
- **Branch:** `swarm/g`. Merges back into `integrations/swarm-p3` at Merge 1 (G + R).

## Scope (from the program plan, Lane G brief)

1. **Ledger first** (this file).
2. **`BackendKind::Cuda` engine arm** — mirror the `Wgpu` arm: device init from the probe, autotune /
   `DeviceLimits` integration (VRAM budgeting per the P2 probe matrix — 4090 = 24 GB discrete), the B3
   lazy device-resident host-boundary contract (host materialization only at the inventoried
   boundaries). Feature-gated under the existing `cuda` cargo feature; **off the default gate**;
   `cargo tree` proof of no default-graph change.
3. **Suites on the 4090** — the full tolerance/parity/det-digest suites under `--features cuda` on the
   pod (`wasm_backend_determinism`, tolerance classes, cross-backend det-digest byte-identity vs CPU).
4. **160M single-host on the 4090 (exit criterion)** — the same 160M llama preset + reference-parity
   harness the P2 gates used, inside the Optimizer tolerance class + a throughput record (tokens/s CUDA
   vs the P2 wgpu figures).
5. **Worker integration smoke** — `daemon-train-worker --features swarm-net,cuda` live WS attach to the
   dev coordinator alongside a local peer, a few rounds, det digests byte-identical (proves the CUDA
   arm rides the live plane).
6. **Autotune** — the §10.5 verdict path picks CUDA when present; budget math uses the probe's numbers
   (discrete: dedicated VRAM; no UMA on this card).

## Ownership / boundaries

- **Own (additive):** `crates/coprocessor/daemon-train/src/{runtime.rs, burn_backend.rs, autotune.rs,
  lib.rs}` (the engine/backend/autotune arm), the worker bin
  `src/bin/daemon-train-worker/{backend.rs, live.rs}` (probe + backend selection + live wiring),
  `daemon-train/Cargo.toml` (lane-owned feature/dep edits), and `daemon-train/tests/*` (the new CUDA
  suites). `.#cuda-train`-related flake additions ONLY if additive + documented (scoped delegated
  rights) — none required this lane (the C3 `.#cuda-train` devShell + staged nvrtc-12.4 are ready).
- **Read-only / off-limits:** other worktrees (p3-r, p3-s, `swarm-p3-integration`), the main checkout,
  FROZEN files (root `Cargo.toml`, `deny.toml`, `flake.nix`), `daemon-swarm-*` crates, `guests/`.

## Frozen surface inherited (extend additively only)

- **`tabi@1` (66 ops) — FROZEN FOREVER.** Lane G adds **no** `tabi` ops (a backend arm is an engine
  choice, not an ABI change). **Wire v42 — FROZEN** (no P3 wire change).
- **`OpBackend` / `TrainerBackend` traits — FROZEN, extended additively only.** The CUDA arm is a new
  `AutodiffBackend` type parameter behind the *unchanged* `BurnBackend<B>` (which already implements
  `OpBackend`); nothing new is added to either trait. `WasmBackend`/`WasmBackendConfig`/`EngineConfig`
  gain **no new fields** — the arm rides the existing `EngineConfig.backend` + `gpu_index` seam.
- **B3 lazy-residency host-boundary inventory (ABI §5.9):** the det lane, scalar/metric readouts,
  `canonical_state_bytes`, `checkpoint_bytes`, `upd_push_tensor`, `grad@1` fold, `MetaReport`.
  `BurnBackend<Autodiff<Cuda>>` inherits the **exact** residency contract the wgpu arm honors (native
  results stay device-resident via `Slot::lazy`; host copies only at the inventoried boundaries) — it
  is a type-parameter swap, so residency is shared code, not re-implemented.
- **Det lane stays host-side CPU fp32 (consensus-invariant).** Every `det_*` op + compression native
  in `BurnBackend` materializes host fp32 (`self.host`) and runs `det_core`, so the consensus digest
  (`canonical_state_bytes` over post-ingest masters) is backend-independent and byte-identical to
  `CpuBackend`. Selecting the CUDA engine changes only the **native** lane (forward/backward/AdamW);
  the det lane and the digest are unchanged by construction (the cross-backend digest tests are the
  guard).

## Design decisions (rationale for choices the brief left to the lane)

### D1 — the arm is a one-line type-parameter swap, exactly like `Wgpu`

`burn_backend.rs` already carries the generic `BurnBackend<B: AutodiffBackend>` with the full lazy
device-residency + det-core-on-host contract. The wgpu arm is the alias
`BurnWgpuBackend = BurnBackend<Autodiff<Wgpu>>`. The CUDA arm mirrors it verbatim:
`BurnCudaBackend = BurnBackend<Autodiff<burn::backend::Cuda>>`, `#[cfg(feature = "cuda")]`. **No op
code is re-implemented** — the tolerance-class native lane, the §5.9 host boundaries, and the
det/consensus lane are the same shared generic. This is the "swap the type parameter with no other
change" seam promise from G1/G2.

### D2 — device init from the probe, mirroring the wgpu match arm

`runtime.rs::HostState::new` gains the single `#[cfg(feature = "cuda")] BackendKind::Cuda` arm: run
the memoized `autotune::probe_cuda()` first (canonical bring-up / availability), then construct
`BurnCudaBackend::with_device(device)` where `device = CudaDevice::new(gpu_index)` (`gpu_index` =
`EngineConfig.gpu_index`, `None` → device 0). This is the exact shape of the `Wgpu` arm
(`probe_wgpu()` then `BurnWgpuBackend::with_device`), keeping selection data-only (no burn type leaks
across the `WasmBackend` seam).

### D3 — VRAM probe via the already-locked `cudarc` (discrete-device honest number)

Unlike wgpu (no total-VRAM query — the wgpu path sources dedicated VRAM from amdgpu sysfs), the CUDA
driver **does** expose total device memory. NVIDIA has no amdgpu-style sysfs, so the honest device
source is the driver API. `autotune::probe_cuda()` (feature `cuda`) queries it via `cudarc::driver`
(`init` → `device::get(ordinal)` → `device::total_mem` + `get_name`), wrapped in `catch_unwind` +
`Result` so it never panics and returns `None` when no device / driver is present (the GPU-skip
convention). This yields `DeviceLimits { vram_mb ≈ 24564, ram_mb, max_alloc_mb, shared_mb: 0,
unified: false }` on the 4090 — a real **discrete** budget, no UMA (§6 verdict path).

- **`cudarc` is a LANE-OWNED, cuda-gated, additive dep — lock-neutral (the C2 dep rule).** `cudarc`
  `0.19.8` is **already resolved** in the committed `Cargo.lock` (pulled by `cubecl-cuda 0.10` under
  the `cuda` feature). daemon-train adds `cudarc = { version = "0.19.8", optional = true,
  default-features = false, features = ["driver"] }` and `dep:cudarc` on the `cuda` feature. The
  `driver` / dynamic-loading features are marker features (`= []`, no dep edges), so **`Cargo.lock`
  stays byte-identical** and `cargo deny` is a no-op (no new third-party crate; same version+source
  already in the graph). Off the default gate exactly like `burn/cuda`. Mirrors C2's target-gated
  `windows` dep precedent. **If adding it changes `Cargo.lock`, that is the "STOP and report" trigger
  from the brief** — verified byte-identical below.
- **`unsafe` gate:** the `total_mem` call is a `cudarc` `unsafe fn`, so the cuda probe module carries a
  scoped `#[allow(unsafe_code)]` — the identical pattern the Windows/macOS FFI probe modules already
  use under the crate's `#![deny(unsafe_code)]` (C2 D1). Every other line still errors on stray
  `unsafe`; the worker bin keeps `#![forbid(unsafe_code)]` and calls only the safe `probe_cuda`
  wrapper.

### D4 — the worker selects CUDA for the *native* lane when present (§10.5 verdict path)

The live worker (`live.rs`) constructed `WasmEngineConfig::default()` (CPU) unconditionally — so a
`--features wgpu` build never actually trained on the GPU in the live path (only the test harnesses
did). Lane G adds a `backend::select_backend()` helper (feature-gated) that, when the `cuda` feature is
built **and** `probe_cuda()` reports a device, returns `(BackendKind::Cuda, gpu_index)`; otherwise it
falls through to the prior default (`Cpu`). This is threaded into `build_wasm_backend` (and the OOM
ladder's rebuild) so the CUDA arm genuinely rides the live plane (Scope 5). **Consensus is
unaffected:** the det lane stays host CPU fp32, so the CUDA peer's post-ingest digests are
byte-identical to CPU peers ingesting the same committed set (the C3 heterogeneity invariant, now with
a GPU-native contributor). The wgpu live path is intentionally left unchanged (not this lane's to
alter); only the additive CUDA branch is wired.

## Additive extensions made (freeze at Merge 1) — exact

### engine (`daemon-train` lib)

- **`runtime.rs`:** `BackendKind::Cuda` variant (`#[cfg(feature = "cuda")]`) + the `HostState::new`
  construction arm (`probe_cuda()` bring-up → `BurnCudaBackend::with_device(CudaDevice::new(idx))`).
  `BackendKind`/`EngineConfig` are otherwise unchanged (same `gpu_index` field the wgpu arm uses).
- **`burn_backend.rs`:** the module `#![cfg]` gains `cuda`; `BurnCudaBackend` alias + `cuda_adapter_available()`
  (delegates to `probe_cuda`). No change to the generic `BurnBackend<B>` impl.
- **`lib.rs`:** `burn_backend` compiled/`pub use`d when `cuda` is on; exports `BurnCudaBackend`,
  `cuda_adapter_available`.
- **`autotune.rs`:** `CudaProbe { gpus, vram_mb, max_alloc_mb, adapter, unified: false }` +
  `probe_cuda()` (memoized, `catch_unwind`, `#[cfg(feature = "cuda")]`) + a `cuda_device_limits`
  mapper (discrete: `vram_mb` from the driver, `shared_mb = 0`, `unified = false`). Unit-tested (the
  mapper is pure; the probe is device-gated).

### worker bin (`daemon-train-worker`)

- **`backend.rs`:** `#[cfg(feature = "cuda")]` branches in `hardware()` (backend lanes `["cuda","cpu"]`,
  discrete VRAM) + `device_limits()` (the CUDA discrete budget), and `select_backend()`.
- **`live.rs`:** `build_wasm_backend` + the ladder rebuild consume `select_backend()` (additive; CPU
  default preserved when the cuda feature/device is absent).

### manifest (`daemon-train/Cargo.toml`)

- `cudarc` optional dep + `dep:cudarc` on the `cuda` feature (D3). Lock-neutral.

## Seams this lane exports (freeze at Merge 1)

- **`daemon_train::BackendKind::Cuda`** (feature `cuda`) + `BurnCudaBackend` + `cuda_adapter_available()`.
- **`daemon_train::autotune::{CudaProbe, probe_cuda}`** (feature `cuda`) + `cuda_device_limits`.
- The worker's cuda-aware `hardware()` / `device_limits()` / `select_backend()`.

## Planned commit slices (each green per the gates; TDD tight test+impl slices)

1. `mirror(G): ledger` — this file.
2. `feat(train): BackendKind::Cuda engine arm + probe + autotune device-limits (green)`.
3. `test(train): cuda tolerance/parity/det-digest + 160M cuda suites (green)`.
4. `feat(train): worker selects the CUDA native lane on the live plane (green)`.
5. `mirror(G): finalize ledger — Merge-1 seams + 4090 results`.

## Gates (Lane G)

`cargo fmt --check` · `cargo clippy --workspace --all-targets -- -D warnings` + the feature combos
(`--features cuda`, `--features swarm-net,cuda` — run in `.#cuda-train` for the cudart build headers) ·
`cargo deny check` (no new root dep; cudarc lock-neutral — STOP-and-report if not) ·
`cargo test --workspace` (Linux host; cuda tests skip cleanly with no device) ·
`cargo run -p xtask -- build-guests` · both wasm32 builds · `typos docs/specs` ·
`cargo tree` proof the default graph is unchanged. On-pod: the cuda suites + 160M parity/throughput +
live attach. Known flake (never modified): the `daemon-conformance` detached-delegation trio —
green-in-isolation.

## Results — (filled at lane close)

_Pending: commit list, 4090 suite results + tolerance evidence, the 160M parity + throughput record,
live-attach smoke evidence, deviations, what Merge-1 must know._
