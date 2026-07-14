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

- **`cudarc` is a LANE-OWNED, cuda-gated, additive dep — NO new crate (the C2 dep rule).** `cudarc`
  `0.19.8` is **already resolved** in the committed `Cargo.lock` (pulled by `cubecl-cuda 0.10` under
  the `cuda` feature). daemon-train adds `cudarc = { version = "0.19.8", optional = true,
  default-features = false, features = ["driver"] }` and `dep:cudarc` on the `cuda` feature. The
  `driver` / dynamic-loading features are marker features (`= []`, no dep edges), so **no new
  crate/version/source enters the graph**. The only `Cargo.lock` change is **exactly one line** — the
  `"cudarc 0.19.8"` edge recorded on the `daemon-train` package node (inherent to declaring any
  optional dep); the `[[package]] cudarc 0.19.8` entry already existed. `cargo deny` is therefore a
  no-op (advisories/bans/licenses/sources are over crates, and no crate is added), and the **default
  dependency graph is unchanged** (`cargo tree -p daemon-train -e normal` shows no cudarc — it is
  pulled only by the off-default `cuda` feature). Mirrors C2's target-gated `windows` dep precedent.
  The brief's "STOP and report" trigger is a **new dep** (new crate/version) — which did **not**
  happen; the single edge line is verified below.
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

### D5 — fat-worker packaging: one binary, probe-ordered graceful degradation (user decision, in-wave)

**User decision (recorded 2026-07-14, mid-wave):** backend packaging is **one fat worker binary** —
ndarray + wgpu + cuda features unioned (cuda target-gated to linux/windows x86_64 at packaging time),
with the **runtime probe selecting the arm**. HARD REQUIREMENT: a cuda-featured worker on a
non-NVIDIA machine must degrade gracefully to wgpu/CPU — no panic, **no link-time dependency**.

Implementation: `select_backend()` (worker `backend.rs`) is the probe-ordered ladder
**CUDA → wgpu → CPU** — each rung taken only when its feature is compiled AND its probe reports a
usable device (plus, for CUDA, the D6 NVRTC readiness gate). Verified evidence:

- **dlopen mode confirmed by construction and by ELF.** cudarc resolves with `fallback-dynamic-loading`
  (+ `cuda-version-from-build-system` from cubecl-cuda), and its `build.rs` emits
  `rustc-cfg=feature="dynamic-loading"` whenever no explicit link mode is chosen — every driver/NVRTC
  symbol goes through lazy `libloading` dlopen. **Nothing in the dep graph forces link-mode cudarc**
  (no `dynamic-linking`/`static-linking` feature reachable). ELF proof: the cuda-featured test binary
  has **no `libcuda`/`libnvrtc` in `DT_NEEDED`** (checked with `readelf -d` on this AMD box).
- **Empirical fallback test, run on this AMD (non-NVIDIA) machine:**
  `cuda_lifecycle::cuda_probe_degrades_cleanly_without_nvidia` — a cuda-featured build probes
  `probe_cuda() == None` (the missing-libcuda dlopen unwind is caught; memoized-consistent),
  `cuda_adapter_available() == false`, `cuda_nvrtc_ready() == false`, **no panic**, and the CPU det
  lane still constructs and `da_build`s. The test intentionally has **no GPU skip** — it asserts the
  fallback path on GPU-less runners and the present path on the 4090.
- The wgpu rung reuses the existing memoized `probe_wgpu()` (its own `catch_unwind`); the wgpu live
  path is thereby also wired (previously the live worker was CPU-only regardless of features).

### D6 — NVRTC strategy: fetch-on-demand runtime dir (user decision, in-wave; fetcher = Merge-1/later)

**User decision (recorded 2026-07-14, mid-wave):** NVRTC is **fetched on demand** (like model
assets), keyed by the **detected driver version**, staged into `DAEMON_CUDA_RUNTIME_DIR` (the
indirection the `.#cuda-train` shell already exports onto `LD_LIBRARY_PATH`); until staged, the probe
**downgrades to Vulkan/CPU**. Lane G does NOT build the fetch machinery — it keeps the runtime-dir
contract clean and gates on readiness:

- **Readiness gate (two legs, both required):** `autotune::cuda_nvrtc_ready()` (memoized,
  `catch_unwind`) requires (1) a **loadable `libnvrtc`** — creates + frees a trivial NVRTC program
  via cudarc, proving `libnvrtc.so.12` dlopens and its symbols resolve — AND (2) the **cudart JIT
  headers**: cubecl-cuda resolves `#include <cuda_runtime.h>` per its `cuda_path()` rule
  (`$CUDA_PATH`, else `/usr/local/cuda` / `/opt/cuda` / `/usr`-with-nvcc) and **panics at
  kernel-compile time** when none resolves — a live-attach smoke on the 4090 caught exactly this
  (worker spawned over bare ssh without `CUDA_PATH`: probe passed, first tensor op panic-spammed;
  the run still finished byte-identical because the det lane is host fp32, but the CUDA lane was
  effectively dead). `cuda_jit_headers_present()` mirrors the search rule non-panicking and requires
  `<cuda_path>/include/cuda_runtime.h`. `select_backend()` requires
  `probe_cuda().is_some() && cuda_nvrtc_ready()` before choosing `BackendKind::Cuda`; anything less
  logs a loud "stage DAEMON_CUDA_RUNTIME_DIR + CUDA_PATH" note and falls through to wgpu/CPU (never
  fails on the first tensor op).
- **What the future fetcher must provide (the contract, from the C2/C3 findings + this lane's runs):**
  a directory (to be exported as `DAEMON_CUDA_RUNTIME_DIR` and prepended to `LD_LIBRARY_PATH` by the
  launcher — the `.#cuda-train` shellHook already does the prepend) containing, at minimum:
  - `libnvrtc.so.12` (+ its `libnvrtc-builtins.so.<ver>`) whose **major.minor matches the box driver's
    CUDA level** (driver 550 ⇒ CUDA 12.4 ⇒ NVIDIA wheel `nvidia-cuda-nvrtc-cu12==12.4.127`). A newer
    nvrtc emits PTX the older driver JIT rejects (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` — the C2 D5
    finding); detect the driver via `cuDriverGetVersion` (or parse `nvidia-smi`), then fetch the
    matching wheel and unpack `nvidia/cuda_nvrtc/lib/*`.
  - a resolvable `libstdc++.so.6` next to nvrtc when the host glibc/libstdc++ is older than the wheel
    expects (the RunPod staging carries `libgcc_s.so.1`; nix-glibc hosts get it from the devShell).
  - it need NOT ship `libcuda`/`libnvidia-nvvm`/`libnvidia-ptxjitcompiler` — those are the box
    **driver's own userspace** and load from the system path (host libcuda under nix glibc is proven,
    C2 D5); the RunPod staging dir includes them only as convenience symlinks.
  - cudart **headers** — the **complete** `include/` tree, not a top-level subset — inside the
    runtime dir, and the launcher must export `CUDA_PATH=$DAEMON_CUDA_RUNTIME_DIR` (in addition to
    the `LD_LIBRARY_PATH` prepend). This is a **runtime** need, not just build-time: cubecl-cuda
    re-includes `cuda_runtime.h` on every NVRTC kernel JIT and panics without it, and the headers
    are self-referential — the first live smoke with a partial tree failed with
    `mma.h(55): catastrophic error: cannot open source file "crt/mma.h"` (the `crt/`,
    `cooperative_groups/`, `cuda/std/` subtrees are required; the complete `cuda_cudart`+cccl
    include set is ~186 entries). Source: the NVIDIA wheel `nvidia-cuda-runtime-cu12`'s `include/`
    (or the nixpkgs `cuda_cudart` + `cuda_cccl` includes) matching the driver level; the pod's
    `/root/cuda-rt-124/include` was completed from the devShell's `cuda-merged-12.9` store path.
  - **Readiness vs downgrade detection:** the fetcher's success criterion IS `cuda_nvrtc_ready()`
    flipping to `true` in a fresh process (dlopen search paths + `CUDA_PATH` are process-start
    state; a worker restarts after staging). No sentinel files, no version parsing at probe time —
    loadability + header presence is the test.

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

## Results — lane close (4090 evidence, deviations, Merge-1 notes)

### Commit list (base `245aef6`, oldest → newest)

| Commit | Subject |
|---|---|
| `e52321c` | `mirror(G): ledger` |
| `d3e3ee1` | `feat(train): BackendKind::Cuda engine arm + probe + autotune device-limits (green)` |
| `82eae25` | `test(train): cuda parity/det-digest/160M suites (green)` |
| `b34dc89` | `feat(train): fat-worker probe-order degradation + nvrtc fetch-on-demand readiness gate (green)` |
| `acc9969` | `fix(train): cuda readiness gate also requires the cudart JIT headers (green)` |
| (this) | `mirror(G): finalize ledger — Merge-1 seams + 4090 results` |

### 4090 suite results (on-box, `/root/daemon-node-p3g` = a clean git clone of `swarm/g`, rebuilt
### on-box at every HEAD — the P2 artifact-drift lesson; worker sha256 fingerprints logged per run)

All run in `.#cuda-train` with `DAEMON_CUDA_RUNTIME_DIR=/root/cuda-rt-124`, HEAD `acc9969`:

- **Full `daemon-train` suite `--features cuda,burn-ndarray`** — green across every target
  (lib 43 + autotune 15, burn_backend_parity 17, guest_lifecycle 9, worker_protocol 4,
  wasm_backend_determinism 15 incl. the new `cross_backend_cuda` trio, cuda_lifecycle 3,
  reference_parity 2, preset_160m 2, abi_surface 2, …). First pass (HEAD `b34dc89`) had ONE failure:
  `worker_protocol::supervisor_probe_assess_join` asserted `gpus == 0` for non-wgpu builds — the
  pre-existing assertion didn't know a cuda build reports a GPU; fixed to admit any GPU-featured
  build (`gpus <= 1`), green on re-run.
- **Cross-backend det-digest byte-identity (the consensus tripwire):**
  `cross_backend_cuda::cross_backend_det_digest_{sparse_loco,diloco,demo}` — 6 rounds each,
  **cpu-vs-CUDA det digests bit-identical every round**, native payloads diverge (tolerance-class),
  both losses fall. The det lane is byte-identical **by construction** (host fp32 det-core shared
  with `CpuBackend`; the CUDA arm never touches it) — these tests are the guard, and they are green
  on the real 4090.
- **Autotune / §10.5 verdict (Scope 6):** `meta_mode_estimates_vs_cuda_probe` green with the real
  probe — `DeviceLimits { vram_mb: 24210, ram_mb: 127935, max_alloc_mb: 24210, shared_mb: 0,
  unified: false }` (the driver-honest discrete numbers; `cuDeviceTotalMem` = 24210 MiB on this
  container ≈ nvidia-smi's 24564 total minus reserve) → tiny-llama fits at micro_batch 64;
  `preset_160m_eligible_on_cuda_discrete` green (160M eligible with margin on the discrete path —
  dedicated VRAM budget, no UMA joint pool). `hardware()` reports
  `backend_lanes ["cuda","cpu"]`, vram 24210, shared 0.
- **Tolerance class:** CUDA joins the wgpu tolerance class — the 160M parity run (below) lands
  final-weight max Δ = 4.746e-6, well inside Optimizer (rtol 2e-4/atol 2e-5); per-step losses were
  byte-equal to the reference on this run. No per-op fixture divergence observed.

### 160M single-host on the 4090 — the exit criterion (headline)

Full record + method + honest interpretation in
[`swarm-p3-throughput.md`](swarm-p3-throughput.md). Summary:

- **Parity (release, 4 steps, TinyStories b=1, matched init):** per-step loss **byte-identical**
  (|Δ| = 0.000e0; 10.846266 → 8.986223), final-weight max Δ = **4.746e-6** (Optimizer class) —
  `loss_parity_within_tolerance_160m_cuda` GREEN.
- **Throughput (release, 3 warmup + 10 measured):** tabi **357.9 tok/s** (2.859 s ± 0.065 s/step),
  reference **933.4 tok/s** (1.096 s ± 0.003 s), ratio **2.61×** — vs the P2 wgpu RADV record
  (tabi 383.9, ref 753.4, 1.96×). The 4090's straight-burn reference is the fastest in the program
  (+24% over RADV); tabi converges to the §5.9 host-residency bound on both boxes (the PCIe-vs-UMA
  analysis in the throughput doc).
- **Preset smoke (dev):** build 8.8 s, 7.0 s/step, `make_update` 46.3 s → 12.5 MB sparse_loco
  payload, loss 10.843 → 9.100 — `preset_160m_trains_on_cuda` GREEN.

### Live-attach smoke (Scope 5) — GREEN, with two real findings en route

Final run: `fleet_heterogeneous_det_lane_agrees` — local Linux peer + the 4090 worker
(`--features swarm-net,cuda`, spawned over `ssh -p 13988 -T`, env
`DAEMON_TRAIN_MODULE=… DAEMON_CUDA_RUNTIME_DIR=/root/cuda-rt-124 LD_LIBRARY_PATH=/root/cuda-rt-124
CUDA_PATH=/root/cuda-rt-124`) against the real Cloudflare dev coordinator
(`https://daemon-swarm-dev.me-dc6.workers.dev`), run **`run-c3-fleet-1784023517`**: worker log shows
**“selecting CUDA native lane (device 0: NVIDIA GeForce RTX 4090, 24210 MiB VRAM)”**, 4 rounds,
**det digests byte-identical every round** (`9fa9a6b0…`, `4446c13e…`, `bf39a25e…`, `1198ea6a…`),
run Finished, zero panics/errors in the transcript. The CUDA engine arm rides the live plane.

The two findings the smoke surfaced (both fixed + recorded):

1. **JIT headers are a runtime requirement (→ `acc9969` + D6).** The first attempt spawned the
   worker without `CUDA_PATH`: the readiness gate (then libnvrtc-only) passed, the worker selected
   CUDA, and cubecl-cuda **panicked on every kernel JIT** (`CUDA installation not found`) — the run
   still finished byte-identical (det lane is host fp32; the round loop tolerated the dead native
   lane on this tiny model), but the CUDA lane was effectively off. Fix: `cuda_nvrtc_ready()` gained
   the header leg (D6), so an unstaged box now downgrades loudly instead of panic-spamming.
2. **The include tree must be complete.** With `CUDA_PATH` set but the staged `include/` partial,
   NVRTC failed with `mma.h(55): catastrophic error: cannot open source file "crt/mma.h"` and the
   pod peer starved (0 rounds committed; local peer 3 — run budget exceeded). Fix: completed
   `/root/cuda-rt-124/include` from the devShell's `cuda-merged-12.9` store path (186 entries incl.
   `crt/`); D6's fetcher contract now requires the full tree.

### Gate matrix (final HEAD `acc9969` + the finalize-ledger commit)

- Host (this box, default devShell): `cargo fmt --check` ✓ · `cargo clippy --workspace
  --all-targets -- -D warnings` ✓ · `cargo deny check` ✓ (advisories/bans/licenses/sources —
  cudarc adds NO new crate) ·   `cargo test --workspace` ✓ except known parallel-load flakes, all on the program's standing
  known-flake list and all **green in isolation** this session: the `daemon-conformance`
  detached-delegation trio member `detached_fanout_materializes_distinct_children` (module 5/5
  isolated; the final `--no-fail-fast` run's ONLY failure), `f1_approvals_pending_is_owner_scoped`
  (1/1 isolated; flaked once early-session), and `drills.rs::late_join_mid_run_syncs_and_contributes`
  (1/1 isolated; flaked once while Lane S's live 160M rehearsal loaded the box). None are swarm-lane
  files this lane touches · `cargo run -p xtask -- build-guests` ✓ · wasm32
  `daemon-swarm-proto` + `daemon-swarm-coordinator` ✓ · `typos docs/specs` ✓.
- Feature combos (`.#cuda-train`): clippy `-p daemon-train --features cuda --all-targets` ✓ ·
  `--features swarm-net,cuda` ✓ · `--features swarm-net,cuda,wgpu` (the fat-worker union) ✓ —
  all `-D warnings`.
- Dep proof: default graph unchanged (`cargo tree -p daemon-train -e normal` — no cudarc/burn-cuda/
  cubecl-cuda); `Cargo.lock` delta = exactly one edge line (`"cudarc 0.19.8"` on the daemon-train
  node; the `[[package]]` entry pre-existed via cubecl-cuda). **No new root dep** — the brief's
  STOP trigger did not fire.
- On-pod (4090): the cuda suites green (above); 160M parity + throughput green; live smoke green.

### Deviations (recorded honestly)

1. **`worker_protocol` assertion widened** (`gpus == 0` → GPU-featured builds assert `gpus <= 1`):
   a test file this lane didn't originally own but whose assertion was factually outdated by the
   cuda lane; one-line, behavior-preserving for every existing build shape.
2. **The wgpu rung of `select_backend` also lights up the wgpu live path** (previously the live
   worker always ran CPU even when built `--features wgpu`). Additive and probe-gated; consensus
   unaffected (det lane host fp32). Called out for Merge-1 review since it changes what a
   wgpu-featured live worker *does* (trains its native lane on the GPU).
3. **`max_alloc_mb` on CUDA = total VRAM** (no per-buffer ceiling exists in the driver API the way
   wgpu's `max_buffer_size` does); the verdict's per-tensor gate then only rejects tensors larger
   than the card, which is the honest semantic on CUDA.
4. **The pod's `cuDeviceTotalMem` reports 24210 MiB** (vs nvidia-smi 24564): the driver's usable
   figure; recorded as-is (a true lower bound, the right budgeting number).
5. **First smoke attempt ran with a stale-featured env** (no `CUDA_PATH`) — caught by the run
   transcript, root-caused, fixed as D6 leg 2, re-run clean. No binary drift: every pod run was
   preceded by an on-box rebuild at the exact lane HEAD with a printed sha256.

### What Merge-1 must know

1. **Freeze (additive):** `BackendKind::Cuda` + `BurnCudaBackend` + `cuda_adapter_available()`;
   `autotune::{CudaProbe, probe_cuda, cuda_nvrtc_ready, cuda_device_limits}`; the worker's
   `select_backend()` probe order (cuda → wgpu → cpu) + `DAEMON_TRAIN_BACKEND=cpu` escape hatch.
   `OpBackend`/`TrainerBackend`/`EngineConfig` shapes unchanged; no tabi ops; wire stays v42.
2. **The fat-worker packaging decision (D5) is implemented and verified** — graceful degradation on
   non-NVIDIA proven on the AMD box (`cuda_probe_degrades_cleanly_without_nvidia`, no GPU-skip) +
   ELF `DT_NEEDED` clean (dlopen mode). Nothing in the dep graph forces link-mode cudarc.
3. **The nvrtc fetch-on-demand contract (D6) is documented + gated** (two-leg readiness: loadable
   libnvrtc AND the complete cudart include tree at `CUDA_PATH`); the fetch machinery itself is the
   agreed Merge-1/later item. Launchers must export BOTH `DAEMON_CUDA_RUNTIME_DIR` (LD path) and
   `CUDA_PATH` — consider folding the `CUDA_PATH` export into the `.#cuda-train` shellHook when the
   runtime dir is set (a one-line flake follow-on; NOT made this lane — flake untouched).
4. **No flake edits were needed** (the C3 `.#cuda-train` shell + staged runtime dir sufficed);
   `Cargo.lock` carries the one cudarc edge line; `deny` green.
5. **Lane S / the fleet:** the 4090 peer now contributes a real GPU-native lane on the live plane
   (this lane's smoke is the small-run proof; run `run-c3-fleet-1784023517`). For the 160M fleet
   run, stage the complete include tree + nvrtc per D6 on the pod (already done at
   `/root/cuda-rt-124`) and spawn the worker with the three env vars above.
6. **Known follow-on from Lane R (recorded, no action this wave): exclude local state from
   `checkpoint_save` so cross-peer checkpoints can mean byte-identical.** Lane R found checkpoint
   objects are per-peer byte-divergent because `CheckpointWire` (runtime.rs, this crate) serializes
   **all** persistents — including `class = 0` **local** optimizer state (AdamW m/v), which
   legitimately differs per peer (each peer runs its own native lane). The `class = 1` replicated
   subset + masters + round bases are the cross-peer-identical part; a future slice should either
   split the wire shape (replicated body + local sidecar) or emit a canonical replicated-only
   variant for the resync path. Lane G touched `checkpoint_save`'s callers but NOT its shape —
   nothing here worsens the divergence (the CUDA arm keeps m/v in the same persistent slots the CPU
   backend uses; `restore_checkpoint` stays bit-exact per-peer, which is all §9 resume needs today).
7. **Good-guest ledger (pod):** `/root/daemon-node-p3g` (clean clone + target), `/root/swarm-g*.bundle`,
   `/root/p3g-*.{sh,log}`, and `/root/cuda-rt-124/include` completed (added headers; nothing
   removed). `/root/daemon-node-c3` untouched. All on the ephemeral container's local disk.
