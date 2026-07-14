# Swarm P2 — Lane C2 ledger (platform probes + fleet lanes)

Lane **C2** of the Swarm P2 WAN Program, Wave 2. Worktree
`/home/j/experiments/daemon-worktree/p2-c2`, branch `swarm/c2`, base `4e821cd` (trunk
`integrations/swarm-p2` @ Merge-1). This file is the single source of truth for what C2 landed, the
seams it exports (freeze at Merge 2), every `flake.nix` edit (with rationale, subject to Merge-2
review), and the fleet-validation evidence. Mirror commit: `mirror(C2): ledger`.

## Scope (from the wave brief)

1. Ledger first (this file).
2. **Windows DXGI/D3D12 probe FFI** — `device_limits()` for Windows per the RESOLVED
   `swarm-windows-vram-design.md` §2 mapping; target-gated `windows` dep; fixture tests
   (discrete/UMA/VGM/WARP) run everywhere; MinGW cross-build of the worker; real-5090 validation
   vs Task Manager / `nvidia-smi` / `vulkaninfo`.
3. **macOS Metal probe** — `recommendedMaxWorkingSetSize` + `maxBufferLength` via a small FFI;
   fixture tests + real M4 validation.
4. **M4 devshell fix** — platform-gate the Linux-only devShell package(s) so `nix develop`
   evaluates on aarch64-darwin.
5. **CUDA lane (RunPod)** — honest assessment of burn-cuda vs wgpu-over-Vulkan-with-nixGL; evidence
   from the actual box; implement what's in scope, record the rest for Merge 2.
6. **Fleet smoke** — the three-platform probe matrix measured end-to-end through the real worker.

## Ownership / boundaries

- Own: `crates/coprocessor/daemon-train` autotune/probe modules (`src/autotune.rs`) + the worker
  backend probe (`src/bin/daemon-train-worker/backend.rs`, **additive functions only**).
- **Scoped `flake.nix` rights** (delegated by the integration owner, ADDITIVE lane outputs only):
  new devShells/packages for the macOS eval fix + any CUDA lane stanza. Every edit documented below
  with rationale, subject to Merge-2 review. Existing outputs untouched.
- Do NOT touch other lanes' worktrees, root `Cargo.toml`, `deny.toml`, or the worker bin's
  transport/`JoinRun` (A3's this wave).

## Frozen surface inherited (extend additively only)

`daemon_train::autotune::DeviceLimits { vram_mb, ram_mb, max_alloc_mb, shared_mb, unified }` is
frozen at P1 Merge 2 (`swarm-p1-ledger.md` seam 2). C2 **does not change the struct** — the
per-platform probes fill the existing five fields. `DedicatedSystemMemory` (Windows) is carried as
a log/telemetry note, not a new struct field, to keep the frozen shape intact.

---

## Decisions (rationale)

### D1 — `#![forbid(unsafe_code)]` → `#![deny(unsafe_code)]` on the daemon-train lib

The platform probes are inherently `unsafe` FFI (DXGI/D3D12 COM on Windows; the Objective-C runtime
+ `sysctlbyname` on macOS). The lib crate root carried `#![forbid(unsafe_code)]`, which **cannot**
be overridden by an inner `#[allow]`. Minimal honest change: relax to `#![deny(unsafe_code)]` and
put `#[allow(unsafe_code)]` on the two cfg-gated FFI modules only (`#[cfg(windows)]`,
`#[cfg(target_os = "macos")]`). Every other line of the crate still errors on stray `unsafe`. The
worker **bin** keeps `#![forbid(unsafe_code)]` untouched — it only calls safe wrappers
(`autotune::probe_windows_device_limits()` / `probe_macos_device_limits()`). All the decision logic
lives in **pure** mapping functions (no `unsafe`, fixture-tested everywhere); the FFI modules only
gather raw scalars into an intermediate struct and hand it to the pure mapper.

### D2 — probes live in `daemon-train` (the crate I own), not a new crate

Keeps the change inside the declared ownership (autotune/probe). The `windows` dep is target-gated
in `daemon-train/Cargo.toml` (`[target.'cfg(windows)'.dependencies]`, consuming the Wave-0 reserved
root `windows = "0.62"` entry — a lane-owned feature/target edit, no root change). macOS uses raw
`extern` FFI (`#[link(name=…, kind="framework")]` to Metal/Foundation + libSystem `sysctlbyname`) —
**no new dependency**.

### D3 — worker probe readout via `DAEMON_TRAIN_PROBE=1`

Fleet validation needs the worker's own `Probe`/assess path to print its `Hardware` +
`DeviceLimits`. Constructing a length-framed CBOR `Command::Probe` by hand across cmd.exe/ssh is
fragile, so `main.rs` gains a tiny additive early-exit: if `DAEMON_TRAIN_PROBE` is set, print
`hardware()` + `device_limits()` (+ the raw adapter scalars) and exit before the stdio loop. This is
additive, does not touch transport/`JoinRun`, and is the exact same `backend::` code the live
`Probe` command runs.

---

## flake.nix edits (ADDITIVE — Merge-2 review)

All under the delegated scoped rights; commit `build(nix): darwin-eval fixes + windows probe lane`.

1. **`devShells.default`: `bubblewrap` gated to Linux** (`lib.optionals
   pkgs.stdenv.hostPlatform.isLinux`). Rationale: bubblewrap is Linux-only
   (`meta.platforms`), and its unconditional presence made `nix develop` *refuse to evaluate* on
   aarch64-darwin — the exact fleet-report M4 blocker. bwrap has no macOS analogue; the
   execute_code sandbox tests already guard on usability. Linux shell contents unchanged.
2. **`devShells.default`: the llama.cpp prebuilt convenience (`LLAMA_PREBUILT_DIR`,
   `LLAMA_PREBUILT_SHARED`, `LD_LIBRARY_PATH`) gated to Linux** (`lib.optionalAttrs … isLinux`).
   Rationale: after the bubblewrap fix the M4 shell still could not *realize* — `llamaCpp` is a
   Linux-shaped derivation whose postInstall hard-fails on darwin ("FATAL: libmtmd.so missing";
   cmake emits `.dylib`), observed on the real M4 (27min failed realize). Darwin has its own Metal
   lanes (`daemon-infer-metal` via `buildEngineWorker`), which don't use the prebuilt. Linux env
   vars unchanged.
3. **`packages.daemon-train-probe-windows` (NEW output)**: the `daemon-train-probe` bin cross-built
   for `x86_64-pc-windows-gnu` via the existing `craneLibWindows` toolchain/env (same
   `windowsCommonArgs` as `daemon-windows`), default features. This is the deployable DXGI
   validation artifact for the 5090 box (never build on-box).

**No existing output was removed or restructured**; edits 1–2 are platform gates on the default
devShell (Linux behavior identical), edit 3 is a new package output.

## Seams exported (freeze at Merge 2)

- Per-platform `device_limits()` sources: Windows DXGI/D3D12 FFI, macOS Metal FFI, existing Linux
  sysfs. Pure mappers `autotune::{windows_device_limits, macos_device_limits}` + the intermediate
  raw structs.
- flake lane additions (macOS eval fix; MinGW `daemon-train-worker-windows`; any CUDA stanza).
- The CUDA-vs-Vulkan RunPod decision + evidence.

---

## Decisions (continued)

### D4 — the full worker bin does NOT link under MinGW (recorded blocker, not C2's)

`cargo check -p daemon-train --bin daemon-train-worker` is **green** for `x86_64-pc-windows-gnu`
(wasmtime 46, burn 0.21, `windows` 0.62 all cross-compile), but the final link fails:
`undefined reference to _invoke_watson` from the `crash_handler` object — `daemon-telemetry`'s
always-on `sentry-rust-minidump` → `crash-handler` native-minidump path references a UCRT symbol
the mingw-w64 msvcrt import lib does not export. That is a **daemon-telemetry / toolchain** issue
(substrate crate, not C2-owned). Mitigation this wave: the new `daemon-train-probe` bin (no
telemetry/stdio stack, same `autotune` probe code) links, ships, and validated on the real 5090.
**Merge-2 item:** decide whether `daemon-telemetry`'s minidump path gets target-gated
(`cfg(not(all(windows, target_env = "gnu")))`) or the worker grows a `no-crash-reporting` feature —
either unblocks a true `daemon-train-worker.exe`.

### D5 — RunPod backend adjudication: **CUDA lane** (Vulkan-via-wrapper REJECTED with evidence)

Investigated on the actual box (`ssh -p 13988 root@213.173.109.230`, RTX 4090, driver 550.127.05,
Ubuntu 22.04 container, nix 2.34.8 single-user):

- **Vulkan is dead-ended by glibc, not by the ICD json.** Reproduced the fleet report, then went
  further with a scoped host-driver dir (`/root/nvidia-vk`: symlinks to the host's
  `libGLX_nvidia.so.0` + its NVIDIA + X11 deps, plus a rewritten ICD manifest) and
  `VK_DRIVER_FILES`/`LD_LIBRARY_PATH`: the loader then *finds and loads* the ICD, but
  `vk_icdGetInstanceProcAddr` fails — `LD_DEBUG=libs` pins the root cause:
  `libnvidia-glcore.so.550.127.05: undefined symbol: __malloc_hook/__free_hook/__memalign_hook/
  __realloc_hook (fatal)` (+ `ErrorF`). The 550-series **GLX-based** ICD needs glibc's removed
  malloc hooks, which the host Ubuntu 22.04 glibc still compat-exports but **nix glibc 2.4x does
  not**. A nixGL-style wrapper around *host* libs cannot fix a missing-symbol-vs-nix-glibc
  mismatch; the real nixGL route would pull nixpkgs' **unfree `nvidia_x11` userspace pinned to the
  container's exact kernel-module version** (550.127.05) — fragile on an ephemeral pod (driver
  changes on redeploy), unfree, and heavier than the CUDA path. **Rejected.**
- **CUDA works under nix glibc.** `cuInit(0) == 0`, device `NVIDIA GeForce RTX 4090`, 24210 MiB —
  called through nix-built python/ctypes loading the *host* `libcuda.so.1` (the driver's CUDA
  stack has no glibc-compat problem).
- **burn-cuda compiles on the box with ZERO lock changes.** Scratch checkout `/root/c2-scratch`
  (tree of `swarm/c2`), `cuda = ["burn/cuda"]` feature: `cargo check -p daemon-train --features
  cuda` **green in 34s**, and `Cargo.lock` **byte-identical** before/after (burn-cuda 0.21 /
  cubecl-cuda 0.10 / cudarc 0.17.8 were already resolved in the committed lock) — so per the
  ledger's dep rule this is a **lane-owned feature edit, NOT a new-root-dep request**.
- **Runtime end-to-end green:** `daemon-train-probe --features cuda` brings up the real burn-cuda
  backend and runs `(t+t).sum() = 12` on the 4090. Runtime requirements discovered en route
  (recorded for the `.#cuda` devshell design): host `libcuda` + `libnvidia-ptxjitcompiler`/
  `libnvidia-nvvm`; **`libnvrtc` matching the driver's CUDA level** (container ships none; nixpkgs
  has dropped ≤12.5, and nvrtc 12.6 emits PTX the 12.4 driver rejects with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` — the working combination was NVIDIA's pip wheel
  `nvidia-cuda-nvrtc-cu12==12.4.127`); cudart **headers** for NVRTC's `#include <cuda_runtime.h>`
  (`CUDA_PATH` env; nixpkgs `cuda_cudart` includes work); and a `libstdc++.so.6` resolvable next
  to nvrtc. All staged under `/root/cuda-rt-124` (temp dir, good-guest).
- **Merge-2 request (flake, integration owner or delegated C2):** a `.#cuda` devShell stanza =
  unfree-scoped `cudaPackages_12_x.cuda_nvrtc` (version keyed to the box's driver) + `cuda_cudart`
  headers + a wrapper exporting `CUDA_PATH`/`LD_LIBRARY_PATH` incl. host driver libs. NOT added
  this wave: nixpkgs-unstable has dropped the 12.4-matching nvrtc, so an honest stanza needs
  either a driver upgrade on the pod or a pinned nvrtc source — an integration-owner trade-off.
  The cargo-side `cuda` feature IS landed (lock-neutral).

## Landed (commits on `swarm/c2`, base `4e821cd`)

| Commit | Subject |
|---|---|
| `0944957` | `feat(train): Windows DXGI/D3D12 + macOS Metal device_limits probes (C2)` |
| `6a3955e` | `build(nix): darwin-eval fixes + windows probe lane (C2, scoped flake rights)` |
| `c443934` | `fix(core): darwin mode_t portability in contained set_mode (C2 cross-lane one-liner)` |
| `f1dabfc` | `feat(train): cuda lane feature + probe runtime smoke (C2 RunPod lane)` |
| (this)   | `mirror(C2): ledger` |

**Cross-lane one-liner flag (Merge-2 review):** `c443934` touches
`crates/engine/daemon-core/src/exec/contained.rs` (ONE line + comment) — outside C2's file
ownership, required because `daemon-core` rides into `daemon-train` via
`daemon-swarm-net → daemon-egress` and its `Mode::from_bits_retain(u32)` does not compile on
darwin (`mode_t` is u16 there), which blocked the M4 deliverable outright. Behavior on Linux is
bit-identical (no-op cast); `daemon-core` exec tests re-run green on Linux. No restructuring.

## Fleet validation evidence (the three-platform probe matrix, measured)

### Windows Server 2022 + RTX 5090 (`37.230.134.194`) — REAL numbers, closes the design checklist

`daemon-train-probe.exe` (MinGW cross-built, scp-deployed, 340 KB static) — raw DXGI/D3D12:

| Probe field | Value | Cross-check |
|---|---|---|
| `DedicatedVideoMemory` | 33 753 661 440 B = **32 190 MiB** | nvidia-smi `memory.total` = 32 607 MiB (Δ = WDDM/OS reserve — expected); **WMI `AdapterRAM` = 4 293 918 720 = the u32-cap trap**, probe immune |
| `DedicatedSystemMemory` | 0 | design: usually 0 |
| `SharedSystemMemory` | 68 660 008 960 B = **65 479 MiB** | ≈ ½ of RAM (130 958 MiB) — the classic DXGI convention |
| `ARCHITECTURE1.UMA` / `CacheCoherentUMA` | **false / false** | discrete, queried not inferred |
| `QueryVideoMemoryInfo(LOCAL).Budget` | 32 948 355 072 B = **31 422 MiB** | ≈ 0.976 × VRAM — the WDDM budget **Task Manager's GPU tab shows as "Dedicated GPU memory"** (same source, per design §1; SSH box, no interactive TM session — nvidia-smi + WMI + vulkaninfo triangulate the same WDDM numbers) |
| `QueryVideoMemoryInfo(NON_LOCAL).Budget` | 67 854 702 592 B = **64 711 MiB** | ≈ ½ RAM; telemetry-only, contributes 0 |
| `GlobalMemoryStatusEx.ullTotalPhys` | **130 958 MiB** | WMI `TotalPhysicalMemory` = 137 320 017 920 B = 130 958 MiB — **exact** |
| WARP skip / WDDM-fallback distrust | exercised | box carries a phantom "Microsoft Basic Display Adapter" (AdapterRAM 0); enumeration picked the 5090 |

→ `DeviceLimits { vram_mb: 32190, ram_mb: 130958, max_alloc_mb: 2047 (DX12 i32::MAX const),
shared_mb: 0 (discrete rule), unified: false }`. Vulkan corroboration: vulkaninfo = RTX 5090,
api 1.4.341, driverInfo 610.74.

### M4 Mac (`62.210.193.129`, Mac16,10, 32 GiB) — Metal probe + devshell fix validated

- **Eval fix:** `nix eval .#devShells.aarch64-darwin.default.drvPath` — was `error: Refusing to
  evaluate package 'bubblewrap-0.11.2'`, now green in **~1.0 s** (warm). Realize: first attempt
  failed in `llama-cpp-prebuilt` (edit #2 above); after the gate the full shell realized (cold
  realize incl. toolchain download ≈ 28 min on this box/network — one-time), subsequent entries
  ~seconds.
- **Build:** `cargo build -p daemon-train --features burn-ndarray -j 5` — green after `c443934`
  (warm incremental; the cold build compiled the full worker tree on the M4).
- **Metal probe (raw):** `recommendedMaxWorkingSetSize` = 26 800 603 136 B = **25 559 MiB**
  (= 78.0 % of RAM on this 32 GiB M4 — NOT the M1's ⅔; the fraction is Metal's own live number,
  which is exactly why the probe reads it instead of hardcoding a ratio),
  `maxBufferLength` = 20 100 448 256 B = **19 169 MiB** (58.5 % RAM; again not the M1's ½),
  `hasUnifiedMemory` = true, `hw.memsize` = 34 359 738 368 (**32 768 MiB**, sysctl cross-check
  exact). → `DeviceLimits { vram_mb: 25559, ram_mb: 32768, max_alloc_mb: 19169, shared_mb: 32768,
  unified: true }`, via BOTH `daemon-train-probe` and the real worker
  (`DAEMON_TRAIN_PROBE=1 daemon-train-worker`, which also printed the full 66-op `Hardware`
  capability report).
- Fixture tests on-box: `cargo test -p daemon-train --lib autotune` **14/14 green** (darwin).

### RunPod RTX 4090 (`213.173.109.230`) — CUDA lane (see D5 for the full adjudication)

- Worker end-to-end (`DAEMON_TRAIN_PROBE=1`, default cpu build): CPU-lane fallback
  `DeviceLimits { vram_mb: 127935, ram_mb: 127935, max_alloc_mb: 0, shared_mb: 0, unified: false }`
  (= MemTotal 131 005 760 kB; by design — no GPU probe in the cpu lane; the 4090's 24 564 MiB
  rides the CUDA lane).
- burn-cuda one-op runtime smoke on the 4090: **`(t+t).sum() = 12`** ✓ (nvrtc 12.4 pip wheel +
  cudart headers + host driver libs, `/root/cuda-rt-124`).
- Fixture tests on-box: **14/14 green** (linux).

### Strix Halo (local) — unchanged Linux sysfs path

The existing sysfs+wgpu probe is untouched (this lane added Windows/macOS branches ahead of it);
the P1-measured numbers (vram 4096 + gtt 120000, unified) remain the Linux row of the matrix, and
the full autotune suite (incl. the new fixtures) runs in the default Linux gate.

### Good-guest ledger (what was left on fleet machines)

- **M4:** `~/daemon-node-c2` (rsync tree + `target/`) — the validation checkout; nix store gains
  (GC-able). Nothing outside it.
- **Windows:** `daemon-train-probe.exe` in the user profile dir (340 KB). Nothing installed.
- **RunPod:** `/root/c2-scratch` (scratch tree + target), `/root/nvidia-vk` + `/root/cuda-rt*`
  (symlink dirs), `/tmp/c2-tree.tar.gz`, `/tmp/nvrtc-ext` + wheel, nix store gains. All on the
  ephemeral container's local disk (nothing on `/workspace`), all deletable.
