# macOS / Apple Silicon UMA GPU-budget findings (autotune `DeviceLimits`)

**Status:** validation notes — untracked, for the `integrations/swarm-p1` (Merge-2) owner to fold
into the unified-memory GPU-budget fix. Do **not** treat as a committed spec change.
**Author/date:** G2 validation pass, 2026-07-13, on real Apple Silicon hardware.
**Scope:** validate/refute the macOS half of the `DeviceLimits { vram_mb, ram_mb, max_alloc_mb,
shared_mb, unified }` design against real Metal/wgpu numbers. The Linux/RADV half (amdgpu sysfs;
`max_buffer_size` clamped to `i32::MAX`) is out of scope here except where Metal contradicts it.

---

## 1. Machine under test

| Property | Value |
|---|---|
| Model | `Macmini9,1` (Apple M1 Mac mini) |
| Chip | Apple M1 (SoC `T8103`), unified memory |
| RAM (`hw.memsize`) | 8 589 934 592 B = **8192 MiB = 8.00 GiB** |
| macOS | 26.3.2 (build 25D2140), Darwin 25.3.0 `xnu-12377.91.3` arm64 |
| Nix | 2.34.7 (daemon `/nix/var/nix/profiles/default`) |
| Toolchain used | nix `nixpkgs#rustc nixpkgs#cargo` (rustc/cargo 1.96.1); Xcode CLT swift 6.3.2 for the Metal probe |

`~/git/` on the Mac holds bare repos `daemon.git`, `daemon-node.git`, `daemon-app.git` (a
daemon-node bare repo **does** exist alongside the superproject). No code needed to be pushed —
a standalone probe crate was the fastest route, so nothing was pushed to any remote.

### sysctl (GPU-relevant)

```
hw.memsize            = 8589934592
hw.model              = Macmini9,1
iogpu.wired_limit_mb  = 0      # AUTO (not admin-overridden)
iogpu.wired_lwm_mb    = 0
iogpu.dynamic_lwm     = 1
```

---

## 2. Probed numbers (the point of the exercise)

### 2a. Metal (`MTLCreateSystemDefaultDevice()`, authoritative) — via `swift metal_probe.swift`

| Metal property | Bytes | MiB | Note |
|---|---:|---:|---|
| `name` | — | — | `Apple M1` |
| `hasUnifiedMemory` | — | — | **true** |
| `isLowPower` / `isHeadless` | — | — | false / false |
| `maxBufferLength` | 4 294 967 296 | **4096.0** | = **exactly 50.0% of RAM** (4 GiB) |
| `recommendedMaxWorkingSetSize` | 5 726 633 984 | **5461.3** | = **exactly 66.7% of RAM** (2/3) |
| `currentAllocatedSize` (idle) | 65 536 | 0.06 | baseline |
| `ProcessInfo.physicalMemory` | 8 589 934 592 | 8192.0 | matches `hw.memsize` |

### 2b. wgpu 29.0.4 (matches the workspace pin: wgpu 29.0.4 / burn 0.21 / cubecl 0.10)

| wgpu field | Bytes | MiB | Note |
|---|---:|---:|---|
| `adapter.get_info().device_type` | — | — | **`IntegratedGpu`** ✅ |
| `.backend` | — | — | `Metal` |
| `.name` | — | — | `Apple M1` |
| `limits().max_buffer_size` | 4 294 967 296 | **4096** | **== Metal `maxBufferLength` exactly** (honest, NOT clamped) |
| `limits().max_storage_buffer_binding_size` | 4 294 967 292 | 4095 | 4 GiB − 4 B |

### 2c. Empirical unified-pool allocation (bounded, best-effort)

Allocated live `STORAGE|COPY_DST|COPY_SRC` buffers in 256 MiB chunks under an `OutOfMemory`
error scope, each verified with a `write_buffer` + `poll(Wait)`, then a real
`copy_buffer_to_buffer` between two live buffers:

```
allocated_total_MiB = 256 512 768 … 3072   (12 × 256 MiB, all succeeded)
copy_ok = true
final_allocated_total_MiB = 3072   live_buffers = 12
```

- Reached **3072 MiB (3.0 GiB)** of simultaneously-live GPU buffers + a working copy — **56.3% of
  `recommendedMaxWorkingSetSize`** (5461 MiB) and **37.5% of RAM**.
- The ceiling was **our self-imposed safety cap (~40% RAM)**, *not* any device limit — allocation
  never OOM'd. This confirms there is **no small "dedicated VRAM" pool**: buffers are served
  straight from unified DRAM, and the pool comfortably exceeds the 2 GiB (`i32::MAX`) figure that
  bounds RADV's clamped `max_buffer_size`. We deliberately did not push toward the 5.3 GiB
  working-set ceiling to avoid swap pressure on an 8 GB machine.

---

## 3. Eligibility arithmetic on THIS Mac

Inputs plugged into the fix design (`unified = true`, `ram_mb = 8192`, GPU budget =
`recommendedMaxWorkingSetSize = 5461 MiB`, `max_alloc_mb = maxBufferLength = 4096 MiB`,
`shared_mb = ram_mb = 8192`). Fixed-VRAM / largest-tensor figures are the autotune `fixed_vram_bytes`
and `max_tensor_bytes` (consistent with spec §VRAM-planning table: 160M ≈ 2.8 GB fixed + acts →
~4.5 GB total / ~2 GB host RAM; 1.2B ≈ 21.6 GB fixed → ~23 GB total / ~15–16 GB host RAM).

### 160M preset — fixed VRAM ≈ 3051 MiB, largest tensor 93 MiB, host RAM ≈ 2048 MiB

| Gate | Check | Result |
|---|---|---|
| max single alloc | 93 ≤ 4096 MiB | ✅ pass (4003 MiB headroom) |
| GPU working set | 3051 + `mb`·act ≤ 5461 MiB | ✅ fits at `mb ≥ 1` (**2410 MiB** headroom for activations) |
| host RAM | 2048 ≤ 8192 MiB | ✅ pass |
| **joint DRAM pool** (unified) | fixed 3051 + host 2048 = **5099** ≤ shared **8192** MiB | ✅ pass (3093 MiB headroom) |

**→ 160M is ELIGIBLE on this 8 GB M1.** (It would be even without the joint check, but the joint
check is what keeps it honest as host RAM grows with peer count.)

### 1.2B row — fixed VRAM ≈ 20.6 GiB (21094 MiB), largest tensor 250 MiB, host RAM ≈ 15–16 GiB

| Gate | Check | Result |
|---|---|---|
| max single alloc | 250 ≤ 4096 MiB | ✅ pass |
| GPU working set | 21094 > 5461 MiB | ❌ **short by 15633 MiB** |
| host RAM | ~15360 > 8192 MiB | ❌ short by ~7168 MiB |
| joint DRAM pool | 21094 (fixed alone) > 8192 MiB | ❌ short by 12902 MiB |

**→ 1.2B is INELIGIBLE on this 8 GB M1** (correctly — a 1.2B fp32-master run cannot fit 8 GB of
unified memory, working-set budget or not).

---

## 4. Recommended macOS `device_limits()` sources (design: validated, lightly amended)

| `DeviceLimits` field | Recommended macOS source | Value here | Notes |
|---|---|---:|---|
| `unified` | wgpu `adapter.get_info().device_type == IntegratedGpu` | **true** | Matches Metal `hasUnifiedMemory`. Heuristic validated on Apple Silicon. |
| `ram_mb` | `sysctl hw.memsize` / MiB | 8192 | Full physical RAM. |
| `vram_mb` | **Metal `recommendedMaxWorkingSetSize`** / MiB (objc/FFI to `MTLCreateSystemDefaultDevice`) | 5461 | **Use the working-set number, NOT 0.** On unified there *is* a real GPU budget = the working set. Not wgpu-queryable → needs a tiny Metal FFI call. Fallback if FFI unavailable: `⌊2/3 · ram_mb⌋` (≈ 5461); a flat 70% overshoots the real value by ~273 MiB (+5%), so prefer 2/3 or the live number. |
| `shared_mb` | `= ram_mb` (the unified physical pool) | 8192 | The pool CPU+GPU jointly draw from. Drives the joint check so `fixed_vram + host_ram` is validated against one pool instead of summed as if VRAM and RAM were disjoint. |
| `max_alloc_mb` | Metal `maxBufferLength` / MiB — **or** wgpu `limits().max_buffer_size` (they agree exactly on Metal) | 4096 | Per-allocation ceiling only, **not** a capacity proxy. On Metal, wgpu's value is trustworthy (see surprise #1), so no Metal FFI is strictly required for this field. |

`iogpu.wired_limit_mb`: present on macOS 26.3.2 but **`0` = auto/kernel-default** on a stock box —
**not usable as a budget source**. It only carries a number when an admin has overridden it
(`sudo sysctl iogpu.wired_limit_mb=N`). Recommendation: **ignore it unless nonzero**; never use `0`
as a budget. `recommendedMaxWorkingSetSize` is the authoritative GPU budget.

**Minimal implementation path for the worker:** one Metal FFI call (`objc2`/`objc` `msg_send` to
`recommendedMaxWorkingSetSize` and `maxBufferLength` on `MTLCreateSystemDefaultDevice()`), plus
`hw.memsize` via `sysctl`. `device_type == IntegratedGpu` sets `unified`. No new heavy dep — the
worker already brings up a wgpu/Metal adapter, so the adapter is in hand; only the two Metal
scalars need an FFI shim (they are absent from wgpu's surface).

---

## 5. Surprises / deltas vs the design hypothesis

1. **wgpu `max_buffer_size` is HONEST on Metal.** It reports the true `maxBufferLength` (4 GiB =
   4 294 967 296 B), **not** clamped to `i32::MAX` as on Linux/RADV. So the "`max_buffer_size` is a
   useless capacity proxy" conclusion is **Vulkan/RADV-specific, not universal**. On macOS,
   `max_alloc_mb` can be sourced directly from wgpu. (It remains a per-allocation ceiling, never a
   total-VRAM figure — the design's separation of `max_alloc_mb` from `vram_mb` still holds.)
2. **`recommendedMaxWorkingSetSize` = exactly 2/3 (66.7%) of RAM** on this 8 GB M1 — squarely
   inside the hypothesized 65–75% band, at the low end. Use the live number where possible; the 70%
   fallback is a slight over-estimate.
3. **`maxBufferLength` = exactly 50% of RAM (4 GiB).** The per-allocation ceiling scales with
   installed memory (Metal documents ≥256 MB and RAM-scaled); here it lands at RAM/2. A largest
   tensor above ~4 GiB would be rejected — none of the current presets approach that (160M: 93 MiB,
   1.2B: 250 MiB).
4. **`recommendedMaxWorkingSetSize` is genuinely not wgpu-exposed** — confirmed; wgpu gives only
   `max_buffer_size`. The design's "needs a Metal-side source" assumption holds.
5. **`device_type` is `IntegratedGpu`** as predicted — the `unified` heuristic is safe on Apple
   Silicon. `hasUnifiedMemory == true` corroborates.
6. **`iogpu.wired_limit_mb` is `0` (auto)** on a stock macOS 26.3.2 — a weaker signal than hoped;
   only meaningful when admin-overridden. Not a substitute for `recommendedMaxWorkingSetSize`.

**Net:** the macOS design is **validated with one amendment** — `vram_mb` must be the Metal
working-set number (not 0, not `max_buffer_size`), `max_alloc_mb` can come from wgpu on Metal
(honest there), `shared_mb = ram_mb` drives the joint unified-pool check, and `iogpu.wired_limit_mb`
should be ignored unless nonzero.

---

## 6. Exact commands used

Recon:
```
ssh m1@51.159.120.241 'uname -a; sw_vers; sysctl hw.memsize hw.model; sysctl iogpu; \
  xcode-select -p; which swift clang xcrun; ls -la ~/git/'
```

Metal probe (`~/tmp/swarm-probe/metal_probe.swift`, run with `xcrun swift metal_probe.swift`):
reads `maxBufferLength`, `recommendedMaxWorkingSetSize`, `hasUnifiedMemory`, `physicalMemory`.

wgpu probe (`~/tmp/swarm-probe/`, `Cargo.toml` pins `wgpu = "29"`, `pollster = "0.4"`):
```
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
export CARGO_HOME="$PWD/.cargo" CARGO_TARGET_DIR="$PWD/target"
nix shell nixpkgs#cargo nixpkgs#rustc --command cargo build --release
./target/release/swarm-probe
```
The crate prints `adapter.get_info()` (device_type/backend/name), `limits().max_buffer_size` +
`max_storage_buffer_binding_size`, then allocates 12×256 MiB live buffers + a copy under an
`OutOfMemory` error scope. wgpu-29 API notes for whoever ports this into the worker:
`Instance::new(InstanceDescriptor::new_without_display_handle_from_env())` (by value);
`DeviceDescriptor` needs `experimental_features: ExperimentalFeatures::disabled()`;
`device.poll(PollType::wait_indefinitely())`; error scopes are `let g = push_error_scope(..);
g.pop().await`.

---

## 7. Cleanup / what was left on the Mac

- **Everything is contained under `~/tmp/swarm-probe/`** (crate source, isolated `.cargo/` registry,
  `target/`, `metal_probe.swift`). Nothing was written outside it.
- **Nix store additions** (shared, harmless, cached): the `nixpkgs#rustc`/`nixpkgs#cargo` closure
  (~2 GiB) pulled by `nix shell` into `/nix/store`, plus the nixpkgs-unstable flake eval cache.
  These are normal Nix artifacts, GC-able with `nix-collect-garbage`.
- **No host tools installed**, no `~/.cargo` (used a scoped `CARGO_HOME`), **no pushes to any git
  remote**, no changes to `~/git/`, no sysctl writes, no daemon/app state touched.
- To fully remove the probe: `ssh m1@51.159.120.241 'rm -rf ~/tmp/swarm-probe'` (left in place for
  reproducibility unless you want it gone).
