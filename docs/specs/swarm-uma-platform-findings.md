# UMA / GPU-memory-budget platform findings (autotune `DeviceLimits`) — platform matrix

**Status:** validation notes — untracked, for the `integrations/swarm-p1` Merge-2 owner to fold
into the unified-memory GPU-budget fix. Do **not** treat as a committed spec change.
**Author/date:** G2 follow-up investigation, 2026-07-13.
**Scope:** the one-stop platform matrix for the `DeviceLimits { vram_mb, ram_mb, max_alloc_mb,
shared_mb, unified }` fix design: **Linux/amdgpu measured** (this file, §1 — Strix Halo),
**macOS measured** (by reference: `swarm-macos-uma-findings.md`, real M1 mini), **Windows
desk-researched** (this file, §2 — wgpu-hal 29.0.4 source-grounded, no hardware).

Prior context: the G2 lane ledger (`swarm-ledger-g2.md`) and the first unified-memory
investigation established that `Hardware.vram_mb = max_buffer_size/MiB = 2047` on this machine is
wgpu-hal's Linux/Mesa `i32::MAX` clamp (`wgpu-hal-29.0.4/src/vulkan/adapter.rs:1437-1447`), not a
memory quantity, and wrongly rejects the 160M preset (needs ~3.3 GiB) and the 1.2B row (~22 GiB)
although both fit the ~121 GiB unified pool (empirical GTT-spill test passed at 4.8 GiB live).

---

## 1. Linux / amdgpu (Strix Halo) — measured

### 1a. Live numbers (2026-07-13, desktop session running)

**Vulkan heaps with VK_EXT_memory_budget** (vulkaninfo, RADV Mesa 25.2.6, GPU0
`Radeon 8060S Graphics (RADV GFX1151)`):

| Heap | Flags | Size | Budget (live) | Usage (this proc) |
|---|---|---:|---:|---:|
| 0 | *(none — host-visible)* | 40.40 GiB | **39.60 GiB** | 0 |
| 1 | `DEVICE_LOCAL` | 80.79 GiB | **79.20 GiB** | 0 |
| **Σ** | | **121.19 GiB** | **118.80 GiB** | |

(The llvmpipe fallback adapter presents 1 heap = 121.50 GiB ≈ MemTotal; ignore it.)

**amdgpu sysfs** (`/sys/class/drm/card1/device/`):

| File | Value |
|---|---:|
| `mem_info_vram_total` | 4096 MiB |
| `mem_info_vis_vram_total` | 4096 MiB (all VRAM CPU-visible — full-BAR, trivially so on an APU) |
| `mem_info_gtt_total` | 120000 MiB (117.19 GiB) |
| `mem_info_vram_used` (live) | ~2201 MiB (desktop/compositor) |
| `mem_info_gtt_used` (live) | ~249 MiB |
| `/proc/meminfo` MemTotal | 124419 MiB (121.49 GiB) |

### 1b. How sysfs maps onto the Vulkan heaps (what RADV is doing)

The heap sizes are **not** the physical carve-out split. Observed arithmetic:
`vram_total + gtt_total = 4096 + 120000 = 124096 MiB = 121.19 GiB = heap0 + heap1`, split
**exactly ⅔ / ⅓**: heap 1 (`DEVICE_LOCAL`) = 80.79 GiB = ⅔ × 121.19; heap 0 (host-visible) =
40.40 GiB = ⅓. On APUs, RADV treats the whole DRAM pool (carve-out + GTT) as GPU-addressable —
the same physical LPDDR5X either way — and *advertises* two-thirds of the combined pool as a
DEVICE_LOCAL heap so that applications sizing themselves by "VRAM heap" behave sensibly, keeping
one-third as an explicitly host-visible heap for upload-style memory types. This is the
resizable-BAR/smart-access-memory presentation taken to its APU limit: "device-local" is a
*placement preference* the kernel (amdgpu/TTM) satisfies from carve-out or GTT transparently, not
a distinct physical region — which is exactly why our empirical 4.8 GiB allocation ran fine with
`gtt_used` at 6.4 GiB. The per-heap **budget** (VK_EXT_memory_budget) is size minus what the rest
of the system currently uses (~2.4 GiB desktop here), updated dynamically.

### 1c. GTT-limit provenance: CONFIGURED, not default

`/sys/module/amdgpu/parameters/gtt_size` is empty, but `/proc/cmdline` carries an explicit
**`amdgpu.gttsize=120000`** plus `ttm.pages_limit=30000000 ttm.page_pool_size=30000000`
(30 M × 4 KiB ≈ 114.4 GiB TTM ceiling). So the 120000 MiB GTT is a **deliberate kernel-cmdline
configuration** on this box (tuned for GPU LLM work), not a kernel default — the historical
default is ~½ of system RAM (would be ~62 GiB here). The worker probe must therefore *read*
`mem_info_gtt_total` rather than assume any fraction of RAM.

### 1d. The Linux analogue of `recommendedMaxWorkingSetSize`

There is no single OS-blessed scalar like Metal's. The closest analogues, in order of fidelity:

1. **Σ heapBudget over all heaps (VK_EXT_memory_budget)** — live ≈ **118.8 GiB** here. This is
   the driver's own usage-aware "allocate up to this without falling off a cliff" number — the
   true analogue. **Not reachable through wgpu** (no budget query API in wgpu 29;
   `MemoryBudgetThresholds` is write-only config); reaching it directly needs `ash` +
   VK_EXT_memory_budget = a new root dependency (integration-owner decision).
2. **`vram_total + gtt_total` (sysfs)** = 121.19 GiB — the static ceiling the budgets converge to
   on an idle system. Reachable today with plain file reads in the worker bin.
3. `MemTotal` = 121.49 GiB — the outer bound (GTT is capped below it by ttm).

### 1e. Validation of the Merge-2 design numbers (90% GTT discount)

Under the fix design, `device_limits()` on this box reports `vram_mb = 4096`,
`shared_mb = 120000`, effective GPU budget = `4096 + 0.9 × 120000 = 112096 MiB ≈ 109.5 GiB`.

| Number | GiB |
|---|---:|
| Design effective budget (vram + 90% GTT) | **109.47** |
| Driver's live Σ heapBudget (idle desktop) | 118.80 |
| Static ceiling (vram + gtt) | 121.19 |

**Verdict: validated, comfortably conservative.** The static 90% discount sits ~9 GiB *below* the
driver's own idle budget — the right side to err on, because (a) the driver budget is dynamic and
shrinks under desktop/compositor load while ours is static, and (b) on a unified machine the
worker's own host-RAM working set (fp32 masters, staging — §5.1) comes out of the same DRAM, which
the design's **joint pool check** (`gpu_bytes + host_ram_bytes ≤ pool`) covers. No amendment
needed; optionally the discount could be stated as
`min(vram + 0.9·gtt, Σ heapBudget when queryable)` if the ash/VK_EXT_memory_budget dependency ever
lands, but sysfs + 90% is honest and sufficient for admission-grade estimates. All three model
rows re-checked against 109.47 GiB: tiny-llama ✓, 160M ✓ (needs ~3.3 GiB + ~2 GiB host, joint ~5.3
GiB), 1.2B ✓ (~22 GiB + ~16 GiB host, joint ~38 GiB) — matching the driver-budget answer, whereas
today's code rejects the latter two.

### 1f. macOS numbers (by reference — measured on a real M1 mini, 8 GiB)

Full detail in `swarm-macos-uma-findings.md`. Summary row for the matrix:
`device_type = IntegratedGpu` ✓; `max_buffer_size = 4096 MiB` = Metal `maxBufferLength` **exactly**
(honest — the `i32::MAX` clamp is Linux/Android-Mesa-specific); working-set analogue =
`recommendedMaxWorkingSetSize` = **5461 MiB = exactly ⅔ of 8192 MiB RAM** (`iogpu.wired_limit_mb=0`
auto — note the same ⅔ fraction RADV uses for its DEVICE_LOCAL heap); not exposed by wgpu, so the
macOS `device_limits()` source is `⅔ × hw.memsize` (sysctl) as the GPU budget with the joint pool
check, and the eligibility arithmetic there admits 160M and correctly rejects 1.2B on 8 GiB.

---

## 2. Windows — desk research (wgpu-hal 29.0.4 source) + the repo's REAL Windows surface

**Correction (2026-07-13, second pass):** an earlier draft concluded "no Windows lane or CI exists
→ post-program". That premise was false. The verified Windows surface:

### 2a′. Verified Windows coverage in the repos

- **daemon-node MinGW cross lane** (`x86_64-pc-windows-gnu` via `pkgsCross.mingwW64` from the
  logos-co `mingw-integration` nixpkgs fork): `flake.nix:24,446-491`. Shipped package outputs:
  `daemon-windows` + `daemon-cli-windows` (`flake.nix:518-519`, built `-p daemon -p daemon-cli`),
  `daemon-infer-llama-windows` (`flake.nix:749`) and `daemon-infer-mistralrs-windows`
  (`flake.nix:821`), plus `llama-cpp-windows` with **CPU + Vulkan** compute backends
  (`flake.nix:879-881`). Interactive lane: `devShells.windows-cross` (`flake.nix:1005-1022`).
- **Release CI cross-builds Windows on every release:** superproject
  `.github/workflows/release.yml:64-100` — `windows` job (ubuntu runner, no Windows host) builds
  the NSIS installer (`package-nsis`) and stages `daemon-<ver>-win64.exe`.
- **Wine E2E exists but is manual/best-effort, not a gate:** superproject `justfile:153-163`
  (`package-windows`: "verify on real Windows — wine is unreliable here"; `smoke-windows`) and
  superproject `flake.nix:460-628` (`apps.smoke-windows`: silent NSIS install into a throwaway
  WINEPREFIX, `daemon.exe --version`, app↔daemon named-pipe flow, uninstaller). daemon-node
  `flake.nix:485`: "Windows artifacts cannot run in the linux build sandbox; wine smoke is
  manual." daemon-app additionally ships `checks.windows-sanity` (PE/import-floor gate — wine
  deliberately excluded) and the `windows-smoke-wine` wine package the smoke reuses.
- **Gap that matters here:** `daemon-train` (the training worker) is **not in the Windows package
  matrix** — only `daemon`, `daemon-cli`, and the two `daemon-infer` workers are cross-built. The
  node binary carries no daemon-train dep by design (`bins/daemon/Cargo.toml:62`). burn supports
  Windows (wgpu→DX12/Vulkan, CUDA), and the infer lane proves Vulkan-on-MinGW works in this tree,
  so there is no known blocker — but a `daemon-train-windows` flake lane does not exist yet.
- Per-PR CI (`ci.yml`) is Linux-only; Windows appears in release CI + manual smoke only.

**What Wine can and cannot validate for the UMA fix:** Wine implements DXGI/D3D12 (vkd3d-proxy)
and reports its *own emulated* memory numbers derived from the host GPU — so a Wine run exercises
compilation, probe code paths, and error handling, but the budget figures it returns are **not**
real Windows numbers (`QueryVideoMemoryInfo` under Wine reflects Wine's internal accounting over
the host Vulkan driver). Wine CI/smoke therefore validates *plumbing*, never *calibration*; the
DXGI budget semantics below still need one real-Windows manual verification.

### 2a. What the wgpu probe reports per backend (source-grounded)

wgpu on Windows prefers **DX12** by default; Vulkan is selectable (vendor ICDs, no Mesa clamp).

### 2a. What the probe would report per backend

| Field | DX12 | Vulkan-on-Windows |
|---|---|---|
| `device_type` | `DXGI_ADAPTER_FLAG_SOFTWARE` → `Cpu` (WARP); `D3D12_FEATURE_DATA_ARCHITECTURE.UMA` → **`IntegratedGpu`**; else `DiscreteGpu` (`dx12/adapter.rs:196-204`) | Vulkan `physical_device_type` mapping (`vulkan/adapter.rs:2128`) — `INTEGRATED_GPU` → `IntegratedGpu` |
| `max_buffer_size` | **`i32::MAX` unconditionally** — "Dx12 does not expose a maximum buffer size in the API. This limit is chosen to avoid potential issues with drivers should they internally store buffer sizes using 32 bit ints" (`dx12/adapter.rs:891-894`) | `min(maintenance4.maxBufferSize, maintenance3.maxMemoryAllocationSize, 2^52)` — the Linux/Mesa `i32::MAX` clamp does **not** apply (`cfg!(target_os = "linux") || cfg!(android)`), so the number is the **vendor driver's honest value** |
| Budget query | **Not exposed.** wgpu-hal *internally* calls `IDXGIAdapter3::QueryVideoMemoryInfo` (LOCAL / NON_LOCAL segment groups) to enforce `MemoryBudgetThresholds` (`dx12/device.rs`, `dx12/suballocation.rs`), but there is no public read API | **Not exposed** (same as Linux — no VK_EXT_memory_budget query surface in wgpu 29) |

Consequences for the three Windows cases:

1. **Discrete GPU (DX12).** `device_type = DiscreteGpu` ✓, but `max_buffer_size` is the same
   2047 MiB constant we mis-used on Linux — a 24 GiB card would report `vram_mb = 2047` under the
   *current* probe. Same failure mode as RADV; the fix design (stop using max-alloc as capacity)
   covers it. The budget truth lives in DXGI: `QueryVideoMemoryInfo(LOCAL).Budget` ≈ ~90% of
   dedicated VRAM, `(NON_LOCAL).Budget` ≈ ~½ of system RAM (the classic DXGI shared-system-memory
   convention) — LOCAL → `vram_mb`, NON_LOCAL → `shared_mb` (with `unified = false`, spill to
   NON_LOCAL is PCIe-speed, so a *steeper* spill discount or spill-excluded default is right for
   discrete: recommend `shared_mb` contributing 0 by default on discrete, budget = LOCAL only).
2. **Integrated iGPU / UMA (DX12).** `UMA.as_bool()` → `IntegratedGpu` ✓ — the `unified`
   classification works unmodified. Budget: `QueryVideoMemoryInfo(LOCAL)` on UMA adapters reports
   the shared-pool budget (OS-managed, typically ~½ RAM); joint pool check applies as on
   Linux/macOS.
3. **Vulkan-on-Windows.** `max_buffer_size` is honest (vendor `maintenance3/4`; e.g. AMD
   Adrenalin ≈ VRAM-scale, NVIDIA ≈ 4 GiB-scale `maxMemoryAllocationSize`) — usable as the real
   per-tensor gate. Capacity/budget still not queryable through wgpu; same DXGI (or
   VK_EXT_memory_budget via ash) story as above.

### 2b. Windows VRAM/UMA detection mechanism — **RESOLVED** (Merge 3)

**RESOLVED (2026-07-13, Merge-3 fold-in).** The concrete detection mechanism is settled by the
user's verified Windows research and written up as the standalone probe design
[`swarm-windows-vram-design.md`](swarm-windows-vram-design.md) (API shapes verified against the
`windows` crate 0.62 docs). The winning mechanism: **DXGI `IDXGIAdapter4::GetDesc3`** for static
sizes, **D3D12 `CheckFeatureSupport(ARCHITECTURE1).UMA`** as the *authoritative* unified flag
(which is exactly what wgpu's DX12 `device_type` heuristic is already derived from —
`wgpu-hal-29.0.4/src/dx12/adapter.rs:196-204` — so there is no tension, the FFI just adds
`CacheCoherentUMA` + Vulkan-on-Windows correctness), and **`IDXGIAdapter3::QueryVideoMemoryInfo`**
(LOCAL/NON_LOCAL) for the dynamic OS budget — the Windows analogue of Metal
`recommendedMaxWorkingSetSize` / VK `heapBudget`, completing the three-platform budget symmetry
table in that design. The `DeviceLimits` field mapping, the trap rules (WMI `AdapterRAM` u32 cap;
WARP/WDDM-fallback distrust; AMD "Variable Graphics Memory" reporting), the §10.5 budget-change
governor hook, the P2 implementation sequence, and an example probe program are all in that file.
The candidate table below is retained as the evaluation record that led to the decision.

Candidate mechanisms that were evaluated (the DXGI/D3D12 trio above won):

| Candidate | What it gives | Reachability | Findings (fill in) |
|---|---|---|---|
| wgpu `adapter.get_info().device_type` | UMA-vs-discrete classification (DX12: `D3D12_FEATURE_DATA_ARCHITECTURE.UMA`; Vulkan: driver `physical_device_type`) | **works today, no new dep** | — |
| `DXGI_ADAPTER_DESC.DedicatedVideoMemory` / `SharedSystemMemory` | static VRAM + shared-pool sizes | `windows` crate FFI (target-gated root dep) | — |
| `IDXGIAdapter3::QueryVideoMemoryInfo` (LOCAL / NON_LOCAL segment groups) | **dynamic OS budgets** — the closest Windows analogue of `recommendedMaxWorkingSetSize` / VK heapBudget | `windows` crate FFI; wgpu-hal uses it internally (`dx12/device.rs`, `suballocation.rs`) but exposes no read API | — |
| `D3D12_FEATURE_DATA_ARCHITECTURE{,1}.UMA` / `CacheCoherentUMA` | authoritative UMA flag (what wgpu's classification is built on, `dx12/adapter.rs:200-201`) | via FFI if finer detail than `device_type` is needed | — |
| `D3DKMTQueryStatistics` (gdi32 kernel-thunk) | per-adapter segment sizes/usage without a D3D device | undocumented-ish FFI, works from any process | — |
| VK_EXT_memory_budget via vendor Vulkan (ash) | per-heap budgets, same shape as Linux §1d | `ash` root dep; unifies the Linux + Windows budget story under one mechanism | — |
| Wine behavior for whichever is chosen | emulated numbers only — plumbing validation | free in existing `smoke-windows` | — |

Non-open parts (hold regardless of the chosen mechanism): `unified` from wgpu `device_type`;
`max_alloc_mb` from wgpu `limits().max_buffer_size` (a **wgpu-enforced** ceiling — buffer creation
validates against it — so the DX12 2047 MiB constant is the correct per-tensor gate on that
backend even though it is artificial); `ram_mb` from `GlobalMemoryStatusEx`/sysinfo-equivalent;
discrete GPUs default `shared_mb`'s contribution to the GPU budget to **0** (NON_LOCAL spill is
PCIe-speed, unlike UMA); no-FFI fallback = unknown-budget sentinel that must not *reject* on VRAM.

**Where it lands (revised given the real CI surface):** the platform-neutral `DeviceLimits`
fields + fixture unit tests for the three Windows cases land at **Merge 2** with the rest of the
UMA fix (nothing Windows-specific blocks it). The Windows probe *plumbing* is **near-program**
work, not "post-program": Windows is a shipped release target with a proven Vulkan-on-MinGW lane,
so the natural sequence is (1) the user's open mechanism research above concludes, (2) a
`daemon-train`/worker Windows flake lane is added next to the infer workers (integration-owner
flake change; no known blocker), (3) the probe FFI lands target-gated behind whatever mechanism
won, (4) `smoke-windows` grows a `daemon-train-worker.exe` probe step (plumbing-only, per the Wine
caveat). Steps 2–4 fit the P2 WAN-gate preparation (where mixed-OS peers become a stated goal)
rather than the P1 Merge-3 gate, whose exit criteria are Linux/Vulkan. **Honest test story:** unit
tests over `DeviceLimits` fixtures (real-Windows numbers once the research produces them; clearly
labeled fixture-based), Wine smoke for plumbing only, one manual real-Windows verification
checklist item — we ship no per-PR Windows test CI, and Wine budget numbers must never be treated
as calibration data.

---

## 3. Platform matrix (summary)

| | Linux/amdgpu APU (measured) | macOS Apple Silicon (measured) | Windows DX12 discrete (desk) | Windows DX12 UMA (desk) | Windows Vulkan (desk) |
|---|---|---|---|---|---|
| `device_type` | IntegratedGpu ✓ | IntegratedGpu ✓ | DiscreteGpu ✓ | IntegratedGpu ✓ | per driver ✓ |
| `max_buffer_size` | **2047 MiB = i32::MAX clamp (artificial)** | 4096 MiB = maxBufferLength (**honest**) | **2047 MiB constant (artificial)** | 2047 MiB constant | vendor value (**honest**) |
| Working-set analogue | Σ VK heapBudget ≈ 118.8 GiB (≈ vram+gtt = 121.2 GiB static) | recommendedMaxWorkingSetSize = ⅔ RAM | DXGI LOCAL Budget ≈ 0.9 × VRAM | DXGI LOCAL Budget ≈ ½ RAM | DXGI / VK_EXT_memory_budget |
| Queryable via wgpu 29? | no | no | no | no | no |
| Probe source (worker) | **sysfs** vram/gtt (no new dep) | **sysctl** ⅔ × hw.memsize (no new dep) | **OPEN** (§2b candidates; likely DXGI FFI) | **OPEN** (§2b) | **OPEN** (§2b; ash is a unifying option) |
| 90%-GTT/⅔-RAM budget validated? | ✓ (109.5 vs driver 118.8 GiB — conservative) | ✓ (⅔ exact match) | pending mechanism research | pending | pending |
| CI coverage today | per-PR (Linux) + `.#vulkan` | manual (real M1 measured) | release cross-build + manual Wine smoke (plumbing only — emulated budgets) | same | same |
