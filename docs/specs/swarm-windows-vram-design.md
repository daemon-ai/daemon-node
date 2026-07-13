# Windows VRAM/UMA probe — resolved design (autotune `DeviceLimits`)

**Status:** design record — **committed at Merge 3** by the integration owner (folds together with
`swarm-uma-platform-findings.md` §2b, which lane M2 committed this wave; that section is now marked
RESOLVED with a pointer here). This is a **design/P2-prep record, not a landed implementation** —
the probe FFI is P2 WAN-gate work (§4), not P1 Merge-3 code.
**Provenance:** the "Windows VRAM/UMA detection mechanism — OPEN" section
(`swarm-uma-platform-findings.md` §2b) is hereby **RESOLVED** by the user's personal research
(2026-07-13), API shapes verified against the `windows` crate **0.62** docs. This file adopts that
research as authoritative and turns it into the probe design.

Sources (user-verified):
- `IDXGIAdapter4::GetDesc3` / `DXGI_ADAPTER_DESC3` —
  <https://learn.microsoft.com/windows/win32/api/dxgi1_6/nf-dxgi1_6-idxgiadapter4-getdesc3>
- `IDXGIAdapter3::QueryVideoMemoryInfo` / `DXGI_QUERY_VIDEO_MEMORY_INFO` —
  <https://learn.microsoft.com/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo>
- `D3D12_FEATURE_DATA_ARCHITECTURE1` (`UMA`, `CacheCoherentUMA`) —
  <https://learn.microsoft.com/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture1>
- `RegisterVideoMemoryBudgetChangeNotificationEvent` —
  <https://learn.microsoft.com/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-registervideomemorybudgetchangenotificationevent>
- `GlobalMemoryStatusEx` —
  <https://learn.microsoft.com/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex>

---

## 1. The three memory questions and their one-true-API answers

| Question | API | Field semantics (the user's decision rules) |
|---|---|---|
| Hardware sizes (static) | DXGI `IDXGIAdapter4::GetDesc3` | `DedicatedVideoMemory` = physical VRAM on discrete (correct > 4 GB); `DedicatedSystemMemory` = BIOS carve-out some iGPUs reserve (usually 0); `SharedSystemMemory` = **ceiling** on borrowable system RAM (~½ installed RAM) — a limit, NOT usage. Never sum dedicated + shared and call it "VRAM". |
| UMA vs discrete (authoritative) | D3D12 device + `CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1)` | `UMA = true` on Intel iGPU / AMD APU / Windows-on-Arm SoC; `false` on discrete. `CacheCoherentUMA` = coherent cache hierarchy (zero-copy friendly). **Queried, not inferred** — do NOT guess from "small dedicated + large shared". |
| Live availability (dynamic) | `IDXGIAdapter3::QueryVideoMemoryInfo` (node 0, per segment group) | `Budget` = what the OS currently grants this process; `CurrentUsage` = what it uses. On UMA adapters everything reports as `LOCAL` (NON_LOCAL ≈ 0). Budget fluctuates with system pressure; exceeding it → intermittent freezes / allocation failures. Same WDDM-sourced numbers as **Task Manager's GPU tab** → trivial manual cross-verification. |

**Traps (adopted verbatim as probe rules):** WMI `Win32_VideoController.AdapterRAM` is `uint32`
(caps at ~4 GB) — never use. Registry `HardwareInformation.qwMemorySize` (QWORD) is the
trustworthy *legacy* cross-check only; DXGI supersedes it. On hybrid (Optimus-style) laptops
inspect **each adapter independently** — never aggregate across adapters. Skip the software
rasterizer (WARP / `DXGI_ADAPTER_FLAG3_SOFTWARE`) during enumeration. Numbers are only trustworthy
on a WDDM 2.0+ driver — the "Microsoft Basic Display Adapter" fallback lies. **AMD "Variable
Graphics Memory" (Ryzen AI Max / Strix Halo on Windows)** can present tens of GB of unified RAM as
`DedicatedVideoMemory` — report the configured allocation and the physical unified RAM as two
separate numbers, never conflate them.

## 2. Resolved `device_limits()` mapping for Windows

| `DeviceLimits` field | Source | Notes |
|---|---|---|
| `unified` | **`ARCHITECTURE1.UMA`** via the D3D12 FFI | This *replaces-and-validates* the wgpu heuristic. NB: wgpu's `device_type == IntegratedGpu` on DX12 is already derived from the very same `D3D12_FEATURE_DATA_ARCHITECTURE.UMA` bit (`wgpu-hal-29.0.4/src/dx12/adapter.rs:196-204`), so on DX12 the "heuristic" is in fact **exact** — no tension, the FFI adds `CacheCoherentUMA` + the confidence of querying the documented feature struct directly (and keeps `unified` correct if the worker ever probes over Vulkan-on-Windows, where wgpu's mapping comes from the vendor driver's `physical_device_type` instead). |
| `vram_mb` | `GetDesc3().DedicatedVideoMemory` | Plus `DedicatedSystemMemory` carried **separately** (a `dedicated_system_mb` note field / log line, usually 0) — not folded into `vram_mb`. On Ryzen-AI-Max-style Variable Graphics Memory this is the *configured allocation*; log physical RAM beside it. |
| `shared_mb` | **UMA adapter:** `min(SharedSystemMemory, QueryVideoMemoryInfo(LOCAL).Budget)` — on UMA everything is LOCAL, so the live LOCAL budget *is* the shared-pool grant; `SharedSystemMemory` caps it statically. **Discrete adapter:** `min(SharedSystemMemory, QueryVideoMemoryInfo(NON_LOCAL).Budget)` — recorded for telemetry, but contributing **0 to the effective GPU budget by default** (PCIe-speed spill), consistent with the program's existing discrete-spill rule. |
| `max_alloc_mb` | wgpu `limits().max_buffer_size` — **unchanged** | On DX12 this is the `i32::MAX` constant (`dx12/adapter.rs:891-894`); it is wgpu-*enforced* at buffer creation, so it stays the per-tensor gate and never a capacity number. |
| `ram_mb` | `GlobalMemoryStatusEx().ullTotalPhys` | |

Verdict behavior (platform-neutral, already in the Merge-2 design): discrete → effective GPU
budget = `vram_mb` (shared contributes 0); UMA → **joint-pool check** with the live Budget as the
working-set ceiling, completing the three-platform symmetry:

### The three-platform budget symmetry table

| Platform | Working-set analogue (the "allocate up to this" number) | Source | Measured/verified |
|---|---|---|---|
| macOS / Apple Silicon | `MTLDevice.recommendedMaxWorkingSetSize` (≈ ⅔ RAM) | Metal (probe: sysctl ⅔ × `hw.memsize`) | measured, M1 mini (⅔ exact) |
| Linux / amdgpu | Σ `heapBudget` over VK heaps (VK_EXT_memory_budget) ≈ vram_total + gtt_total static | vulkaninfo / sysfs (probe: sysfs, 90 % GTT discount — validated conservative) | measured, Strix Halo |
| Windows | `IDXGIAdapter3::QueryVideoMemoryInfo(...).Budget` (LOCAL on UMA; LOCAL = VRAM grant on discrete) | DXGI (probe: `windows` crate FFI) | desk-resolved; one manual Task-Manager cross-check pending |

## 3. Dynamic-budget note (§10.5 governor interplay — design note, NOT P1 work)

`QueryVideoMemoryInfo().Budget` is dynamic, and DXGI provides
`RegisterVideoMemoryBudgetChangeNotificationEvent` for long-running processes. A budget *shrink*
is the Windows analogue of the §10.5 inference-pressure preemption signal: the OS is telling the
training tenant to yield GPU memory. The natural (future) wiring is: budget-change event →
worker re-runs the autotune verdict against the new Budget → if the current micro-batch no longer
fits, treat it as the §10.5 `Throttle{paused}` / OOM-probe path (halve or pause), exactly the
recovery machinery that already exists. Recorded here so the governor design has the hook named;
no P1/P2 gate depends on it.

## 4. Implementation plan (per the corrected sequence — P2 WAN-gate prep)

1. **Dependency:** target-gated `windows = "0.62"` in the frozen root `Cargo.toml`
   (`[target.'cfg(windows)'.dependencies]`, integration-owner change), consumed by the worker bin
   only. Features honestly required by the APIs used:
   - `Win32_Foundation` (HRESULT, BOOL, LUID, HANDLE)
   - `Win32_Graphics_Dxgi` (factory/adapter interfaces, `GetDesc3`, `QueryVideoMemoryInfo`,
     `DXGI_ADAPTER_FLAG3`, segment groups; + `Win32_Graphics_Dxgi_Common` for shared DXGI types)
   - `Win32_Graphics_Direct3D` (`D3D_FEATURE_LEVEL`) and `Win32_Graphics_Direct3D12`
     (`D3D12CreateDevice`, `CheckFeatureSupport`, `D3D12_FEATURE_ARCHITECTURE1`)
   - `Win32_System_SystemInformation` (`GlobalMemoryStatusEx`)
   - `Win32_System_Threading` **only if** the §3 budget-notification event is wired
     (`CreateEventW`) — omit for the probe-only landing.
2. **Ordering:** (a) `daemon-train`/worker joins the Windows flake lane beside the infer workers
   (`daemon-infer-*-windows`, daemon-node `flake.nix:749/821` precedent; no known blocker — the
   Vulkan-on-MinGW infer lane already builds); (b) the probe FFI lands target-gated behind the §2
   mapping; (c) the superproject `smoke-windows` grows a `daemon-train-worker.exe` probe step.
3. **Tests:** fixture-based unit tests over `DeviceLimits` encoding the §1 field semantics —
   discrete (large `DedicatedVideoMemory`, NON_LOCAL budget recorded-but-zero-contribution), UMA
   (LOCAL-only budgets, joint-pool), Variable-Graphics-Memory (huge "Dedicated" on an APU with the
   physical-RAM sibling number), WARP-skipped enumeration, WDDM-fallback distrust. Clearly labeled
   fixture-based.
4. **Wine caveat (unchanged):** Wine's DXGI/D3D12 layer reports emulated numbers over the host
   Vulkan driver — `smoke-windows` validates *plumbing only*, never calibration.
5. **Manual verification (once, real Windows):** cross-check the probe's `GetDesc3` +
   `QueryVideoMemoryInfo` output against **Task Manager's GPU tab** (same WDDM source, per the
   user's decision rules) on one discrete and one UMA machine.

## 5. Appendix — example probe program (desk-verified shape, `windows` 0.62)

**NOT yet compiled on Windows** — signature shapes checked against the windows-crate 0.62 docs
only (`GetDesc3` *returns* the struct; `QueryVideoMemoryInfo` and `D3D12CreateDevice` take
out-pointers). For the eventual `daemon-train-worker` windows module.

```rust
use windows::core::Result;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12Device, D3D12_FEATURE_ARCHITECTURE1,
    D3D12_FEATURE_DATA_ARCHITECTURE1,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter4, IDXGIFactory6, DXGI_ADAPTER_FLAG3_SOFTWARE,
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    DXGI_QUERY_VIDEO_MEMORY_INFO,
};

fn probe() -> Result<()> {
    let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory2(Default::default())? };
    let mut i = 0u32;
    loop {
        // EnumAdapters1 + cast, or EnumAdapterByGpuPreference::<IDXGIAdapter4>; stop on NOT_FOUND.
        let adapter: IDXGIAdapter4 = match unsafe { factory.EnumAdapterByGpuPreference(i, ..) } {
            Ok(a) => a,
            Err(_) => break,
        };
        i += 1;

        let desc = unsafe { adapter.GetDesc3()? };          // 0.62: RETURNS the struct
        if desc.Flags.contains(DXGI_ADAPTER_FLAG3_SOFTWARE) {
            continue; // skip WARP (trap rule)
        }
        let dedicated = desc.DedicatedVideoMemory;           // physical VRAM (discrete)
        let dedicated_sys = desc.DedicatedSystemMemory;      // BIOS carve-out, usually 0
        let shared_ceiling = desc.SharedSystemMemory;        // borrowable-RAM CEILING (~1/2 RAM)

        // UMA: queried, never inferred (out-pointer shapes in 0.62).
        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)? };
        let device = device.expect("D3D12CreateDevice out-pointer");
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default(); // NodeIndex = 0
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&mut arch as *mut D3D12_FEATURE_DATA_ARCHITECTURE1).cast(),
                size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
            )?;
        }
        let unified = arch.UMA.as_bool();                    // authoritative
        let coherent = arch.CacheCoherentUMA.as_bool();

        // Live OS-granted budgets (node 0; UMA => everything LOCAL, NON_LOCAL ~ 0).
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe {
            adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)?;
            adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut non_local)?;
        }

        // DeviceLimits mapping (§2): vram = dedicated; shared = min(ceiling, budget-per-case);
        // discrete contributes shared=0 to the effective GPU budget; UMA uses the LOCAL budget
        // as the working-set analogue in the joint-pool check.
        let _ = (dedicated, dedicated_sys, shared_ceiling, unified, coherent, local, non_local);
    }
    Ok(())
}
```

(Enumeration line intentionally schematic — `EnumAdapterByGpuPreference` generic-arg plumbing is
the one bit to settle when this first compiles on a real Windows toolchain; per-adapter handling
is the hybrid-laptop trap rule.)
