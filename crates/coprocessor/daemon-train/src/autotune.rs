// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! VRAM autotune + OOM-probe micro-batch sizing (G2 — spec §5.1 VRAM planning, §10.5 OOM path,
//! ABI §8 host-side budgets).
//!
//! The MVP's `assess` used a coarse `master_bytes × 3` VRAM guess. G2 replaces it with a real
//! budget: a resource model ([`Autotune`]) derived from the module's [`crate::MetaReport`] fields
//! (`param_bytes`/`master_bytes`/`grad_bytes` + persistents = the fixed cost; `act_bytes_est` = the
//! per-micro-batch activation cost, since the meta pass runs at batch = 1) compared against probed
//! device limits ([`DeviceLimits`]) → an [`AutotuneVerdict`] (eligibility + chosen micro-batch).
//!
//! ## The verdict + the OOM probe (§10.5)
//!
//! [`Autotune::verdict`] is the *analytical* form: start at the largest power-of-two micro-batch
//! ≤ `max_microbatch` and halve until `fixed + micro_batch · act` fits the VRAM budget (floor 1 →
//! ineligible if even a single sequence does not fit). [`probe_microbatch`] is the *runtime* form
//! for the same halving ladder driven by a real trial: on a GPU OOM the worker halves the
//! micro-batch and re-probes (the §10.5 recovery rung, mapped into the trap taxonomy as
//! [`crate::TrapCode::BudgetMemory`] / `daemon_swarm_run::protocol::ErrorClass::OutOfMemory` — see
//! [`oom_error_class`]).
//!
//! ## What wgpu exposes for VRAM probing (honest inventory)
//!
//! wgpu's `Adapter` exposes `get_info()` (name/backend/device type) and `limits()`
//! (`max_buffer_size` — the largest single allocation, a hard per-tensor ceiling). It does **not**
//! expose total or free VRAM: the Vulkan / WebGPU surface has no device-memory-size query. So
//! [`DeviceLimits::max_alloc_mb`] is the one truly device-honest number; total VRAM
//! ([`DeviceLimits::vram_mb`]) is sourced by the caller from the GPU-governor policy cap (§10.5) or
//! the node's effective-resource computation, not from wgpu. See [`WgpuProbe`] (feature `wgpu`).

use crate::meta::MetaReport;

const MIB: u64 = 1 << 20;

/// The default micro-batch ceiling the autotune search starts from (a power of two; the verdict
/// halves down from here to whatever fits VRAM). Chosen so a small preset gets a sensible batch and
/// a large one halves down deterministically.
pub const DEFAULT_MAX_MICROBATCH: u32 = 64;

/// Byte size of a stored element of `dtype` (ABI §3.2), mirroring `runtime::dtype_size`.
fn dtype_size(dtype: u32) -> u64 {
    match dtype {
        3 => 8,     // I64
        1 | 2 => 2, // BF16 / F16
        6 | 7 => 1, // U8 / Bool
        _ => 4,     // F32 / I32 / U32
    }
}

fn numel(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}

/// The largest power of two `<= n` (and `>= 1`).
fn pow2_floor(n: u32) -> u32 {
    if n <= 1 {
        1
    } else {
        1u32 << (31 - n.leading_zeros())
    }
}

/// Probed (or policy-capped) device limits the autotune compares the resource model against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Dedicated VRAM in MiB — the device's own memory (sysfs `mem_info_vram_total` on
    /// Linux/amdgpu; a true lower bound). NOT wgpu-queryable; the worker sources it (see module
    /// docs). On a unified/integrated device this is the small "carve-out", not the usable budget
    /// (which spills into `shared_mb`).
    pub vram_mb: u64,
    /// Effective host RAM in MiB.
    pub ram_mb: u64,
    /// Largest single GPU allocation in MiB (wgpu `max_buffer_size`); `0` = unknown / unbounded
    /// (the CPU / ndarray lanes, or when no device was probed).
    pub max_alloc_mb: u64,
    /// Shared / spillover memory budget in MiB (GTT on Linux/amdgpu — `mem_info_gtt_total`; the
    /// unified-memory pool the device can page tensors into beyond its dedicated VRAM carve-out).
    /// `0` = none (a classic discrete GPU with no host spill). Additive; `Default` = 0 preserves
    /// the pre-UMA behavior.
    pub shared_mb: u64,
    /// Whether the device shares host DRAM (an integrated/unified GPU, or the CPU lane — from
    /// `AdapterInfo.device_type == IntegratedGpu | Cpu`). When set, the verdict treats device +
    /// host footprints as competing for ONE physical DRAM pool (a joint budget), instead of two
    /// independent VRAM / RAM budgets. Additive; `Default` = `false` preserves the discrete path.
    pub unified: bool,
}

/// A resource model of one built module, derived from its [`MetaReport`]. VRAM cost is
/// `fixed_vram_bytes + micro_batch · act_bytes_per_mb` (§8 host-side accounting).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Autotune {
    /// Micro-batch-independent VRAM: params (storage dtype) + fp32 master + fp32 grad + persistents.
    pub fixed_vram_bytes: u64,
    /// Per-micro-batch activation bytes (the meta pass measured this at batch = 1).
    pub act_bytes_per_mb: u64,
    /// CPU-side working set (masters + round base + offloaded persistents + staging), §5.1.
    pub host_ram_bytes: u64,
    /// Per-round payload estimate (bytes).
    pub payload_bytes: u64,
    /// Largest single param master (bytes) — checked against the device's max single allocation.
    pub max_tensor_bytes: u64,
}

/// The autotune verdict: eligibility + the chosen micro-batch + footprint estimates. This is the
/// authoritative G2 verdict type (M1's 160M preset and B3's worker consume it); the frozen
/// `daemon_swarm_run::backend::Assessment` / `protocol::Eligibility` shapes carry a projection of it
/// (the micro-batch rides in `reasons` / `headroom`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutotuneVerdict {
    /// Whether the module fits at micro-batch ≥ 1 under the device limits.
    pub eligible: bool,
    /// The chosen micro-batch (largest power of two ≤ `max_microbatch` that fits); `0` if ineligible.
    pub micro_batch: u32,
    /// VRAM estimate at the chosen micro-batch, in MiB.
    pub vram_mb_estimate: u64,
    /// Host-RAM estimate, in MiB.
    pub ram_mb_estimate: u64,
    /// Per-round payload estimate, in bytes.
    pub payload_bytes_estimate: u64,
    /// Halvings the search applied from the `max_microbatch` power-of-two start to the chosen batch.
    pub oom_retries: u32,
    /// Human-readable reasons (why-not, or informational headroom + chosen micro-batch).
    pub reasons: Vec<String>,
}

impl Autotune {
    /// Build the resource model from a module's [`MetaReport`].
    #[must_use]
    pub fn from_meta(report: &MetaReport) -> Self {
        let persistent_bytes: u64 = report
            .persistent
            .iter()
            .map(|(_, dims, dt, _)| numel(dims) * dtype_size(*dt))
            .sum::<u64>()
            + report
                .det_persistent
                .iter()
                .map(|(_, dims, _)| numel(dims) * 4)
                .sum::<u64>();
        let max_tensor_bytes = report
            .params
            .iter()
            .map(|(_, dims, _)| numel(dims) * 4)
            .max()
            .unwrap_or(0);
        Self {
            fixed_vram_bytes: report.param_bytes
                + report.master_bytes
                + report.grad_bytes
                + persistent_bytes,
            // `act_bytes_est` is the meta pass's activation estimate at batch = 1; floor at 1 byte so
            // the halving search always makes progress even for a degenerate (activation-free) module.
            act_bytes_per_mb: report.act_bytes_est.max(1),
            host_ram_bytes: report.host_ram_bytes_est,
            payload_bytes: report.payload_bytes_est,
            max_tensor_bytes,
        }
    }

    /// A resource model from a raw param layout (name, dims, dtype) — the [`crate::WasmBackend`]
    /// build path, which has the registered params but not a full meta pass. The activation cost is
    /// a coarse proxy (`master_bytes`, i.e. one fp32 copy of the model per sequence); the worker's
    /// meta path ([`Self::from_meta`]) carries the real `act_bytes_est`.
    #[must_use]
    pub fn from_params(params: &[(Vec<u32>, u32)]) -> Self {
        let param_bytes: u64 = params
            .iter()
            .map(|(d, dt)| numel(d) * dtype_size(*dt))
            .sum();
        let master_bytes: u64 = params.iter().map(|(d, _)| numel(d) * 4).sum();
        let max_tensor_bytes = params.iter().map(|(d, _)| numel(d) * 4).max().unwrap_or(0);
        Self {
            fixed_vram_bytes: param_bytes + master_bytes + master_bytes, // storage + master + grad
            act_bytes_per_mb: master_bytes.max(1),
            host_ram_bytes: master_bytes * 2,
            payload_bytes: master_bytes,
            max_tensor_bytes,
        }
    }

    /// The eligibility verdict + chosen micro-batch under `limits`, searching down from the largest
    /// power-of-two micro-batch ≤ `max_microbatch` (§5.1 planning; the §10.5 halving ladder in
    /// analytical form). Floor of 1 → ineligible if a single sequence does not fit.
    ///
    /// ## Unified-memory budget (the UMA fix — spec §5.1/§10.5 amendment)
    ///
    /// The **effective device budget** is `vram_mb + 90% of shared_mb`: dedicated VRAM plus a
    /// documented spill discount on the shared (GTT/unified) pool the device can page into. On a
    /// classic discrete GPU `shared_mb == 0`, so this reduces to `vram_mb` and the path below is
    /// unchanged.
    ///
    /// On a **unified** device (`limits.unified`, e.g. an integrated GPU where VRAM and host RAM
    /// are the SAME physical DRAM), the independent VRAM and host-RAM checks are wrong: they would
    /// double-count one pool and, worse, reject against a driver-clamped `max_buffer_size` VRAM
    /// proxy (2047 MiB on Linux/Mesa) even though the runtime provably spills into GTT. So the
    /// unified path replaces them with a single **joint pool** check: the device footprint plus the
    /// host-RAM footprint must fit together in `min(effective budget, ram_mb)`. The per-buffer
    /// `max_tensor_bytes ≤ max_alloc_mb` gate is kept exactly as-is on both paths (it is a hard
    /// single-allocation ceiling, not a total-memory budget).
    #[must_use]
    pub fn verdict(&self, limits: &DeviceLimits, max_microbatch: u32) -> AutotuneVerdict {
        let ram_budget = limits.ram_mb.saturating_mul(MIB);
        let max_alloc = if limits.max_alloc_mb == 0 {
            u64::MAX
        } else {
            limits.max_alloc_mb.saturating_mul(MIB)
        };
        // Effective device budget: dedicated VRAM + a 90% spill discount on the shared pool.
        let effective_vram_mb = limits
            .vram_mb
            .saturating_add(limits.shared_mb.saturating_mul(9) / 10);
        let effective_vram = effective_vram_mb.saturating_mul(MIB);

        let ineligible = |reason: String| AutotuneVerdict {
            eligible: false,
            micro_batch: 0,
            vram_mb_estimate: self.fixed_vram_bytes.div_ceil(MIB).max(1),
            ram_mb_estimate: self.host_ram_bytes.div_ceil(MIB).max(1),
            payload_bytes_estimate: self.payload_bytes,
            oom_retries: 0,
            reasons: vec![reason],
        };

        // A single param master must fit one GPU buffer (the wgpu-queryable hard ceiling). Kept
        // verbatim on both the discrete and unified paths.
        if self.max_tensor_bytes > max_alloc {
            return ineligible(format!(
                "largest tensor {} MiB exceeds max single allocation {} MiB",
                self.max_tensor_bytes.div_ceil(MIB),
                limits.max_alloc_mb
            ));
        }

        if limits.unified {
            // Joint pool: device + host footprints compete for one physical DRAM (§5.1 UMA).
            let pool = effective_vram.min(ram_budget);
            let start = pow2_floor(max_microbatch.max(1));
            let mut mb = start;
            let mut retries = 0u32;
            loop {
                let device_need = self
                    .fixed_vram_bytes
                    .saturating_add(self.act_bytes_per_mb.saturating_mul(u64::from(mb)));
                let joint_need = device_need.saturating_add(self.host_ram_bytes);
                if joint_need <= pool {
                    return AutotuneVerdict {
                        eligible: true,
                        micro_batch: mb,
                        vram_mb_estimate: device_need.div_ceil(MIB).max(1),
                        ram_mb_estimate: self.host_ram_bytes.div_ceil(MIB).max(1),
                        payload_bytes_estimate: self.payload_bytes,
                        oom_retries: retries,
                        reasons: vec![format!(
                            "fits at micro_batch={mb} (unified pool ~{} MiB: device ~{} MiB + host \
                             ~{} MiB, budget {} MiB = min(vram {} + 90%·gtt {}, ram {})){}",
                            joint_need.div_ceil(MIB),
                            device_need.div_ceil(MIB),
                            self.host_ram_bytes.div_ceil(MIB),
                            pool / MIB,
                            limits.vram_mb,
                            limits.shared_mb,
                            limits.ram_mb,
                            if retries > 0 {
                                format!("; halved {retries}× from {start}")
                            } else {
                                String::new()
                            }
                        )],
                    };
                }
                if mb <= 1 {
                    let device_need = self.fixed_vram_bytes.saturating_add(self.act_bytes_per_mb);
                    return ineligible(format!(
                        "insufficient unified memory even at micro_batch=1: need {} MiB (device {} \
                         + host {}), have {} MiB",
                        device_need
                            .saturating_add(self.host_ram_bytes)
                            .div_ceil(MIB),
                        device_need.div_ceil(MIB),
                        self.host_ram_bytes.div_ceil(MIB),
                        pool / MIB
                    ));
                }
                mb /= 2;
                retries += 1;
            }
        }

        // Discrete path (unchanged behavior; `effective_vram == vram_mb` when `shared_mb == 0`).
        // Host RAM is micro-batch-independent (masters + round base + staging, §5.1).
        if self.host_ram_bytes > ram_budget {
            return ineligible(format!(
                "insufficient host RAM: need {} MiB, have {} MiB",
                self.host_ram_bytes.div_ceil(MIB),
                limits.ram_mb
            ));
        }

        let start = pow2_floor(max_microbatch.max(1));
        let mut mb = start;
        let mut retries = 0u32;
        loop {
            let need = self
                .fixed_vram_bytes
                .saturating_add(self.act_bytes_per_mb.saturating_mul(u64::from(mb)));
            if need <= effective_vram {
                return AutotuneVerdict {
                    eligible: true,
                    micro_batch: mb,
                    vram_mb_estimate: need.div_ceil(MIB).max(1),
                    ram_mb_estimate: self.host_ram_bytes.div_ceil(MIB).max(1),
                    payload_bytes_estimate: self.payload_bytes,
                    oom_retries: retries,
                    reasons: vec![format!(
                        "fits at micro_batch={mb} (~{} MiB VRAM, ~{} MiB host RAM){}",
                        need.div_ceil(MIB),
                        self.host_ram_bytes.div_ceil(MIB),
                        if retries > 0 {
                            format!("; halved {retries}× from {start}")
                        } else {
                            String::new()
                        }
                    )],
                };
            }
            if mb <= 1 {
                return ineligible(format!(
                    "insufficient VRAM even at micro_batch=1: need {} MiB, have {} MiB",
                    self.fixed_vram_bytes
                        .saturating_add(self.act_bytes_per_mb)
                        .div_ceil(MIB),
                    effective_vram_mb
                ));
            }
            mb /= 2;
            retries += 1;
        }
    }
}

/// One step of the runtime OOM probe: did the trial micro-batch fit, or OOM?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStep {
    /// The trial micro-batch ran without an allocation failure.
    Fits,
    /// The trial hit a GPU/host allocation failure (OOM).
    Oom,
}

/// The §10.5 OOM-probe halving ladder driven by a real trial: run at `start`, and on an `Oom` halve
/// and retry until it `Fits` or drops below `floor`. Returns the largest fitting micro-batch, or
/// `None` if even `floor` OOMs (→ ineligible). Deterministic and backend-agnostic, so the halving
/// logic is unit-tested without a GPU (the real wgpu trial wraps a step in `catch_unwind`).
pub fn probe_microbatch(
    start: u32,
    floor: u32,
    mut trial: impl FnMut(u32) -> ProbeStep,
) -> Option<u32> {
    let floor = floor.max(1);
    let mut mb = start.max(floor);
    loop {
        match trial(mb) {
            ProbeStep::Fits => return Some(mb),
            ProbeStep::Oom => {
                if mb <= floor {
                    return None;
                }
                mb = (mb / 2).max(floor);
            }
        }
    }
}

/// The trap taxonomy a GPU allocation failure maps into (§10.5): the worker replaces the instance
/// and re-probes the micro-batch. Recorded here as the single mapping point so the OOM path is
/// discoverable (the runtime already maps a wasmtime "memory" trap to
/// [`crate::TrapCode::BudgetMemory`]; a wgpu allocation panic surfaces at the worker as an
/// [`ErrorClass::OutOfMemory`]).
///
/// [`ErrorClass::OutOfMemory`]: daemon_swarm_run::protocol::ErrorClass::OutOfMemory
#[must_use]
pub fn oom_error_class() -> daemon_swarm_run::protocol::ErrorClass {
    daemon_swarm_run::protocol::ErrorClass::OutOfMemory
}

/// Parse an amdgpu sysfs memory-total file (`mem_info_vram_total` / `mem_info_gtt_total`) into MiB.
///
/// The kernel exposes these as a single decimal **byte** count (e.g. `"4294967296\n"` = 4096 MiB;
/// `"125829120000\n"` = 120000 MiB). Returns `None` on an empty / non-numeric file so the caller
/// can fall back to another source. Pure (no I/O) so the worker's real file read stays a thin
/// wrapper and the parse is unit-tested with fixture strings.
#[must_use]
pub fn parse_amdgpu_mem_mb(contents: &str) -> Option<u64> {
    contents.trim().parse::<u64>().ok().map(|bytes| bytes / MIB)
}

/// A honest snapshot of what a wgpu adapter exposes for resource planning (feature `wgpu`). Total
/// VRAM is **not** wgpu-queryable (see the module docs), so [`Self::vram_mb`] reports the adapter's
/// `max_buffer_size` as a documented lower-bound proxy; [`Self::max_alloc_mb`] is the same limit
/// used as the hard per-allocation ceiling. The GPU-governor policy is the authoritative VRAM
/// budget for eligibility.
#[cfg(feature = "wgpu")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuProbe {
    /// Usable adapters found (1 when a default-device adapter initializes; wgpu adapter enumeration
    /// across all devices needs a direct `wgpu` dep = a frozen-root change, so this is "≥1 usable").
    pub gpus: u32,
    /// `max_buffer_size` in MiB — the largest single allocation (the one device-honest number),
    /// also used as the per-buffer ceiling.
    pub max_alloc_mb: u64,
    /// The adapter name (`get_info().name`, e.g. "AMD Radeon … (RADV …)").
    pub adapter: String,
    /// The graphics backend (`get_info().backend`, e.g. "Vulkan").
    pub backend: String,
    /// The adapter device type (`get_info().device_type` — e.g. "IntegratedGpu", "DiscreteGpu",
    /// "Cpu"). Debug-formatted so no direct `wgpu`-type dependency is needed.
    pub device_type: String,
    /// Whether this is a unified-memory device (`device_type` is `IntegratedGpu` or `Cpu`): the GPU
    /// shares host DRAM, so the autotune verdict uses a joint memory pool (see [`DeviceLimits`]).
    pub unified: bool,
}

/// Probe the default wgpu device for its adapter info + limits (feature `wgpu`). Returns `None`
/// (never panics) when no adapter can be brought up. Reads only wgpu-queryable fields (`get_info`
/// incl. `device_type`, `limits`).
///
/// **Memoized process-wide.** cubecl's `ComputeClient::init` panics with "already registered" if
/// the default device's client is already up, so `probe_wgpu` is the canonical bring-up — call it
/// (or [`crate::wgpu_adapter_available`], which delegates here) before any wgpu tensor work.
///
/// **Register-or-reuse (fragility fix).** The bring-up (`init_setup`) both requests the adapter and
/// registers the client. If a burn tensor op won the race and registered the default device first,
/// `init_setup` panics; rather than caching `None` (which would make a *present* GPU look absent
/// for the rest of the process), the probe recognizes the "already registered" panic as proof an
/// adapter exists and returns a reuse marker (`Some`) so availability stays correct. The worker's
/// assess path always probes first (its meta pass runs on the CPU engine — no prior wgpu op), so it
/// always takes the full tier-1 path and gets the real `device_type` / `max_alloc`.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn probe_wgpu() -> Option<WgpuProbe> {
    use std::sync::OnceLock;
    static PROBE: OnceLock<Option<WgpuProbe>> = OnceLock::new();
    PROBE.get_or_init(probe_wgpu_uncached).clone()
}

#[cfg(feature = "wgpu")]
fn probe_wgpu_uncached() -> Option<WgpuProbe> {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions, WgpuDevice};

    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    // Tier 1 — canonical bring-up: `init_setup` requests the adapter AND registers the default
    // client. Full adapter info incl. `device_type` (→ `unified`).
    let attempt = std::panic::catch_unwind(|| {
        let setup =
            init_setup::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, RuntimeOptions::default());
        let info = setup.adapter.get_info();
        let limits = setup.adapter.limits();
        let max_alloc_mb = (limits.max_buffer_size / MIB).max(1);
        let device_type = format!("{:?}", info.device_type);
        let unified = matches!(device_type.as_str(), "IntegratedGpu" | "Cpu");
        WgpuProbe {
            gpus: 1,
            max_alloc_mb,
            adapter: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type,
            unified,
        }
    });
    std::panic::set_hook(prev);

    match attempt {
        Ok(probe) => Some(probe),
        Err(payload) => {
            // Tier 2 — register-or-reuse: an "already registered" panic means an adapter is up
            // (a burn op registered the default client before this probe). Report availability
            // rather than caching `None`; the per-buffer limit / device_type are unknown via reuse
            // (the worker path never hits this — it probes before any wgpu op).
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("");
            if msg.contains("already registered") {
                Some(WgpuProbe {
                    gpus: 1,
                    max_alloc_mb: 0,
                    adapter: "reused (default client already registered)".to_string(),
                    backend: "wgpu".to_string(),
                    device_type: "Unknown".to_string(),
                    unified: false,
                })
            } else {
                None
            }
        }
    }
}

// =====================================================================================
// Windows DXGI/D3D12 device-memory probe (swarm-windows-vram-design.md §2 mapping).
// The pure mapper + raw struct are unconditional (fixture-tested on every platform); the
// actual DXGI/D3D12 FFI is `#[cfg(windows)]` + target-gated `windows` dep.
// =====================================================================================

/// Static + live memory numbers for one DXGI adapter, gathered by the Windows FFI (or a fixture).
/// All byte counts as reported by the OS; the pure [`windows_device_limits`] mapper turns these
/// into [`DeviceLimits`] per the design's §2 field mapping and its trap rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DxgiAdapterMemory {
    /// `GetDesc3().DedicatedVideoMemory` — physical VRAM on discrete (correct > 4 GiB); on an APU's
    /// "Variable Graphics Memory" this is the *configured allocation*, not physical RAM.
    pub dedicated_video: u64,
    /// `GetDesc3().DedicatedSystemMemory` — BIOS carve-out some iGPUs reserve (usually 0). Carried
    /// as a telemetry note; **never folded into `vram_mb`**.
    pub dedicated_system: u64,
    /// `GetDesc3().SharedSystemMemory` — the static **ceiling** on borrowable system RAM (~½ RAM),
    /// a limit, NOT usage.
    pub shared_system: u64,
    /// `DXGI_ADAPTER_FLAG3_SOFTWARE` — the WARP software rasterizer; skip during enumeration.
    pub is_software: bool,
    /// `D3D12_FEATURE_DATA_ARCHITECTURE1.UMA` — authoritative unified flag (queried, not inferred).
    pub uma: bool,
    /// `D3D12_FEATURE_DATA_ARCHITECTURE1.CacheCoherentUMA` — coherent cache hierarchy (telemetry).
    pub cache_coherent_uma: bool,
    /// `QueryVideoMemoryInfo(node 0, LOCAL).Budget` — the live OS-granted budget for the LOCAL
    /// segment group (on UMA this is the shared-pool grant; on discrete ≈ 0.9 × VRAM, the number
    /// Task Manager's GPU tab shows). Same WDDM source as Task Manager → trivial cross-check.
    pub budget_local: u64,
    /// `QueryVideoMemoryInfo(node 0, NON_LOCAL).Budget` — the live NON_LOCAL budget (≈ ½ RAM on
    /// discrete; ≈ 0 on UMA). Recorded for telemetry; contributes **0** to the discrete GPU budget.
    pub budget_non_local: u64,
}

/// Map one non-WARP DXGI adapter's memory numbers to [`DeviceLimits`] (design §2).
///
/// - `unified` ← `ARCHITECTURE1.UMA` (authoritative; replaces-and-validates the wgpu heuristic).
/// - `vram_mb` ← `DedicatedVideoMemory` (physical VRAM / configured VGM allocation).
/// - `shared_mb`: **UMA** → `min(SharedSystemMemory, LOCAL.Budget)` (on UMA everything is LOCAL, so
///   the live LOCAL budget is the shared-pool grant, statically capped by `SharedSystemMemory`);
///   **discrete** → **0** (NON_LOCAL spill is PCIe-speed and contributes 0 to the effective GPU
///   budget by default, per the program's discrete-spill rule — the NON_LOCAL budget is recorded on
///   [`DxgiAdapterMemory`] for telemetry, not fed to the verdict).
/// - `max_alloc_mb` ← wgpu `max_buffer_size` (passed in; the DX12 `i32::MAX` constant when wgpu is
///   absent) — the per-tensor gate, unchanged.
/// - `ram_mb` ← `GlobalMemoryStatusEx().ullTotalPhys` (passed in).
///
/// Returns `None` for a WARP / software adapter (the caller skips it during enumeration).
///
/// **VGM safety:** on Variable-Graphics-Memory APUs `dedicated_video` can present tens of GB of
/// unified RAM; because the [`Autotune::verdict`] unified path clamps the joint pool to
/// `min(vram + 90%·shared, ram)`, the physical-RAM ceiling caps the inflated VRAM figure — the
/// design's "never conflate configured allocation with physical RAM" rule is enforced by the
/// verdict, and `ram_mb` is the true physical bound.
#[must_use]
pub fn windows_device_limits(
    adapter: &DxgiAdapterMemory,
    ram_mb: u64,
    max_alloc_mb: u64,
) -> Option<DeviceLimits> {
    if adapter.is_software {
        return None; // WARP / software rasterizer — skip (trap rule).
    }
    let shared_mb = if adapter.uma {
        // UMA: the LOCAL budget is the live shared-pool grant; SharedSystemMemory caps it statically.
        adapter.shared_system.min(adapter.budget_local) / MIB
    } else {
        // Discrete: NON_LOCAL spill contributes 0 to the effective GPU budget by default.
        0
    };
    Some(DeviceLimits {
        vram_mb: adapter.dedicated_video / MIB,
        ram_mb,
        max_alloc_mb,
        shared_mb,
        unified: adapter.uma,
    })
}

/// The DX12 per-buffer ceiling wgpu reports (`max_buffer_size`) when the probe has no live wgpu
/// adapter: `i32::MAX` bytes ("Dx12 does not expose a maximum buffer size in the API",
/// `wgpu-hal dx12/adapter.rs:891-894`). In MiB (2047) — a wgpu-enforced per-tensor gate, never a
/// capacity number.
pub const DX12_MAX_BUFFER_MB: u64 = (i32::MAX as u64) / MIB;

// =====================================================================================
// macOS Metal device-budget probe (swarm-macos-uma-findings.md §4 mapping).
// =====================================================================================

/// Metal device scalars gathered by the macOS FFI (or a fixture). All byte counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalAdapterMemory {
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the "allocate up to this" GPU budget (≈ ⅔ RAM on
    /// Apple Silicon); the working-set analogue of the three-platform budget symmetry.
    pub recommended_working_set: u64,
    /// `MTLDevice.maxBufferLength` — the per-allocation ceiling (≈ ½ RAM); honest on Metal (wgpu's
    /// `max_buffer_size` agrees exactly, so this doubles as `max_alloc_mb`).
    pub max_buffer_length: u64,
    /// `sysctl hw.memsize` (== `ProcessInfo.physicalMemory`) — full physical RAM.
    pub phys_ram: u64,
    /// `MTLDevice.hasUnifiedMemory` — Apple Silicon is always true.
    pub has_unified: bool,
}

/// Map Metal device scalars to [`DeviceLimits`] (findings §4): `vram_mb` = the working-set budget
/// (NOT 0, NOT `max_buffer_size`), `shared_mb` = `ram_mb` (the unified physical pool that drives the
/// joint check), `max_alloc_mb` = `maxBufferLength`, `unified` = `hasUnifiedMemory`.
#[must_use]
pub fn macos_device_limits(metal: &MetalAdapterMemory) -> DeviceLimits {
    let ram_mb = metal.phys_ram / MIB;
    DeviceLimits {
        vram_mb: metal.recommended_working_set / MIB,
        ram_mb,
        max_alloc_mb: metal.max_buffer_length / MIB,
        // The unified physical pool CPU+GPU jointly draw from; drives the joint-pool check so
        // `fixed_vram + host_ram` is validated against one pool. On Apple Silicon = physical RAM.
        shared_mb: if metal.has_unified { ram_mb } else { 0 },
        unified: metal.has_unified,
    }
}

// -------------------------------------------------------------------------------------
// Windows FFI (DXGI/D3D12). Compiled only for the Windows target; the `windows` crate is a
// target-gated dep. All decision logic lives in `windows_device_limits` above (fixture-tested
// everywhere); this module only gathers raw scalars.
// -------------------------------------------------------------------------------------
#[cfg(windows)]
#[allow(unsafe_code)]
mod win_ffi {
    use super::{DeviceLimits, DxgiAdapterMemory, DX12_MAX_BUFFER_MB, MIB};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12CreateDevice, ID3D12Device, D3D12_FEATURE_ARCHITECTURE1,
        D3D12_FEATURE_DATA_ARCHITECTURE1,
    };
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, IDXGIAdapter4, IDXGIFactory6, DXGI_ADAPTER_FLAG3_SOFTWARE,
        DXGI_CREATE_FACTORY_FLAGS, DXGI_GPU_PREFERENCE_UNSPECIFIED,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    /// Physical RAM in MiB from `GlobalMemoryStatusEx().ullTotalPhys`; `0` on failure.
    fn ram_mb() -> u64 {
        let mut status = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `status` is a valid, `dwLength`-initialized MEMORYSTATUSEX out-pointer.
        match unsafe { GlobalMemoryStatusEx(&mut status) } {
            Ok(()) => status.ullTotalPhys / MIB,
            Err(_) => 0,
        }
    }

    /// Probe the first non-WARP DXGI adapter → [`DeviceLimits`], plus the raw numbers for logging.
    /// Returns `None` when no usable adapter is found (or DXGI is unavailable).
    pub(super) fn probe() -> Option<(DeviceLimits, DxgiAdapterMemory)> {
        let ram = ram_mb();
        // SAFETY: DXGI factory/adapter/device calls with correctly-typed out-pointers; every result
        // is checked. COM objects are dropped at scope end (windows-crate RAII).
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let mut i = 0u32;
            loop {
                let adapter: IDXGIAdapter4 =
                    match factory.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_UNSPECIFIED) {
                        Ok(a) => a,
                        Err(_) => return None, // exhausted enumeration with no usable adapter
                    };
                i += 1;

                let desc = match adapter.GetDesc3() {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let is_software = (desc.Flags.0 & DXGI_ADAPTER_FLAG3_SOFTWARE.0) != 0;
                if is_software {
                    continue; // skip WARP (trap rule)
                }

                // UMA is queried via a D3D12 device (authoritative), not inferred.
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).is_err() {
                    continue;
                }
                let Some(device) = device else { continue };
                let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
                let _ = device.CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE1,
                    (&mut arch as *mut D3D12_FEATURE_DATA_ARCHITECTURE1).cast(),
                    size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
                );

                let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ =
                    adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local);
                let _ = adapter.QueryVideoMemoryInfo(
                    0,
                    DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                    &mut non_local,
                );

                let raw = DxgiAdapterMemory {
                    dedicated_video: desc.DedicatedVideoMemory as u64,
                    dedicated_system: desc.DedicatedSystemMemory as u64,
                    shared_system: desc.SharedSystemMemory as u64,
                    is_software,
                    uma: arch.UMA.as_bool(),
                    cache_coherent_uma: arch.CacheCoherentUMA.as_bool(),
                    budget_local: local.Budget,
                    budget_non_local: non_local.Budget,
                };
                // wgpu's DX12 `max_buffer_size` is the fixed i32::MAX constant; use it directly (no
                // live wgpu adapter is needed for the probe-only cross build).
                let limits = super::windows_device_limits(&raw, ram, DX12_MAX_BUFFER_MB)?;
                let _ = HANDLE::default(); // (budget-change event handle wiring is §3, not probe-time)
                return Some((limits, raw));
            }
        }
    }
}

/// Probe Windows GPU memory via DXGI/D3D12 → [`DeviceLimits`] (design §2). `None` off Windows or
/// when no usable (non-WARP) adapter is found. Safe wrapper over the `#[cfg(windows)]` FFI.
#[must_use]
pub fn probe_windows_device_limits() -> Option<DeviceLimits> {
    #[cfg(windows)]
    {
        win_ffi::probe().map(|(limits, raw)| {
            eprintln!(
                "daemon-train probe (windows/DXGI): {raw:?} dedicated_system_mb={} \
                 budget_local_mb={} budget_non_local_mb={} -> {limits:?}",
                raw.dedicated_system / MIB,
                raw.budget_local / MIB,
                raw.budget_non_local / MIB,
            );
            limits
        })
    }
    #[cfg(not(windows))]
    {
        None
    }
}

// -------------------------------------------------------------------------------------
// macOS FFI (Metal + libSystem sysctl). Compiled only for macOS. No new dependency — raw `extern`
// FFI to the Objective-C runtime + Metal/Foundation frameworks + libSystem `sysctlbyname`. All
// mapping lives in `macos_device_limits` above (fixture-tested everywhere).
// -------------------------------------------------------------------------------------
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod mac_ffi {
    use super::{macos_device_limits, DeviceLimits, MetalAdapterMemory};
    use core::ffi::{c_char, c_void};

    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }

    /// `[obj selector]` returning an unsigned integer (NSUInteger / u64).
    unsafe fn msg_u64(obj: *mut c_void, sel: *mut c_void) -> u64 {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
            unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    /// `[obj selector]` returning an Objective-C `BOOL` (a signed char on arm64 macOS).
    unsafe fn msg_bool(obj: *mut c_void, sel: *mut c_void) -> bool {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool =
            unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn sel(name: &core::ffi::CStr) -> *mut c_void {
        unsafe { sel_registerName(name.as_ptr()) }
    }

    fn sysctl_u64(name: &core::ffi::CStr) -> u64 {
        let mut val: u64 = 0;
        let mut len = size_of::<u64>();
        // SAFETY: `val`/`len` are valid out-pointers sized for a u64 sysctl scalar.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&mut val as *mut u64).cast(),
                &mut len,
                core::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 {
            val
        } else {
            0
        }
    }

    pub(super) fn probe() -> Option<DeviceLimits> {
        // SAFETY: MTLCreateSystemDefaultDevice returns a valid MTLDevice (or null → bail); the
        // selectors are no-argument accessors returning scalar NSUInteger/BOOL, called via a
        // correctly-typed objc_msgSend. The device is intentionally leaked (probe runs once).
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return None;
            }
            let working_set = msg_u64(device, sel(c"recommendedMaxWorkingSetSize"));
            let max_buffer = msg_u64(device, sel(c"maxBufferLength"));
            let has_unified = msg_bool(device, sel(c"hasUnifiedMemory"));
            let metal = MetalAdapterMemory {
                recommended_working_set: working_set,
                max_buffer_length: max_buffer,
                phys_ram: sysctl_u64(c"hw.memsize"),
                has_unified,
            };
            let limits = macos_device_limits(&metal);
            eprintln!("daemon-train probe (macos/Metal): {metal:?} -> {limits:?}");
            Some(limits)
        }
    }
}

/// Probe macOS GPU budget via Metal (`recommendedMaxWorkingSetSize`/`maxBufferLength`/
/// `hasUnifiedMemory`) + `sysctl hw.memsize` → [`DeviceLimits`] (findings §4). `None` off macOS or
/// when no Metal device is available.
#[must_use]
pub fn probe_macos_device_limits() -> Option<DeviceLimits> {
    #[cfg(target_os = "macos")]
    {
        mac_ffi::probe()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_fixture(
        param_bytes: u64,
        act: u64,
        host_ram: u64,
        max_tensor_dims: &[u32],
    ) -> MetaReport {
        MetaReport {
            abi: 1 << 16,
            params: vec![("w".into(), max_tensor_dims.to_vec(), 0)],
            persistent: Vec::new(),
            det_persistent: Vec::new(),
            param_bytes,
            master_bytes: param_bytes,
            grad_bytes: param_bytes,
            act_bytes_est: act,
            payload_bytes_est: 128,
            ingest_bytes_est: 0,
            host_ram_bytes_est: host_ram,
            op_calls: std::collections::BTreeMap::new(),
            ingest_op_calls_per_peer: 0,
            ops_used: Vec::new(),
            value_dependent: false,
        }
    }

    #[test]
    fn pow2_floor_rounds_down() {
        assert_eq!(pow2_floor(1), 1);
        assert_eq!(pow2_floor(2), 2);
        assert_eq!(pow2_floor(3), 2);
        assert_eq!(pow2_floor(63), 32);
        assert_eq!(pow2_floor(64), 64);
        assert_eq!(pow2_floor(0), 1);
    }

    /// `oom_probe_halves_microbatch`: the halving ladder finds the largest fitting micro-batch.
    #[test]
    fn oom_probe_halves_microbatch() {
        // A device where anything > 2 OOMs: 8 → 4 → 2 (fits), i.e. two halvings.
        let chosen = probe_microbatch(8, 1, |mb| {
            if mb > 2 {
                ProbeStep::Oom
            } else {
                ProbeStep::Fits
            }
        });
        assert_eq!(chosen, Some(2));

        // Fits immediately at the start.
        assert_eq!(probe_microbatch(8, 1, |_| ProbeStep::Fits), Some(8));

        // Even the floor OOMs → ineligible (None).
        assert_eq!(probe_microbatch(8, 1, |_| ProbeStep::Oom), None);

        // Floor is respected (never probes below it): floor 4, everything OOMs → None after 4.
        let mut seen = Vec::new();
        let out = probe_microbatch(16, 4, |mb| {
            seen.push(mb);
            ProbeStep::Oom
        });
        assert_eq!(out, None);
        assert_eq!(seen, vec![16, 8, 4]); // stops at the floor, does not go to 2
    }

    /// The analytical verdict agrees with the runtime probe on the same budget.
    #[test]
    fn verdict_matches_probe_ladder() {
        // fixed = 3 MiB, act = 1 MiB/mb. Budget 6 MiB → fits at mb where 3 + mb ≤ 6 → mb ≤ 3 → 2.
        let a = Autotune {
            fixed_vram_bytes: 3 * MIB,
            act_bytes_per_mb: MIB,
            host_ram_bytes: MIB,
            payload_bytes: 0,
            max_tensor_bytes: MIB,
        };
        let limits = DeviceLimits {
            vram_mb: 6,
            ram_mb: 64,
            max_alloc_mb: 0,
            ..Default::default()
        };
        let v = a.verdict(&limits, 8);
        assert!(v.eligible);
        assert_eq!(v.micro_batch, 2);
        assert_eq!(v.oom_retries, 2); // 8 → 4 → 2

        let probe = probe_microbatch(pow2_floor(8), 1, |mb| {
            if 3 * MIB + u64::from(mb) * MIB <= 6 * MIB {
                ProbeStep::Fits
            } else {
                ProbeStep::Oom
            }
        });
        assert_eq!(probe, Some(v.micro_batch));
    }

    /// `assess_rejects_insufficient_vram`: a device with less VRAM than a single sequence needs is
    /// ineligible; a device below the host-RAM need is ineligible; an oversized tensor is rejected.
    #[test]
    fn assess_rejects_insufficient_vram() {
        // 100 MiB fixed model, 10 MiB/mb activation.
        let a = Autotune::from_meta(&meta_fixture(100 * MIB, 10 * MIB, 50 * MIB, &[16, 16]));

        // Plenty of VRAM + RAM → eligible, largest pow2 ≤ 64 that fits.
        let ok = a.verdict(
            &DeviceLimits {
                vram_mb: 8192,
                ram_mb: 8192,
                max_alloc_mb: 0,
                ..Default::default()
            },
            64,
        );
        assert!(ok.eligible && ok.micro_batch >= 1);

        // Not enough VRAM for even one sequence (fixed 100 + 10 = 110 MiB > 64) → ineligible.
        let no_vram = a.verdict(
            &DeviceLimits {
                vram_mb: 64,
                ram_mb: 8192,
                max_alloc_mb: 0,
                ..Default::default()
            },
            64,
        );
        assert!(!no_vram.eligible);
        assert_eq!(no_vram.micro_batch, 0);
        assert!(no_vram.reasons[0].contains("VRAM"));

        // Enough VRAM but not enough host RAM → ineligible on the RAM gate.
        let no_ram = a.verdict(
            &DeviceLimits {
                vram_mb: 8192,
                ram_mb: 8,
                max_alloc_mb: 0,
                ..Default::default()
            },
            64,
        );
        assert!(!no_ram.eligible);
        assert!(no_ram.reasons[0].contains("RAM"));

        // A single tensor larger than the max allocation → ineligible on the alloc ceiling.
        let big = Autotune::from_meta(&meta_fixture(100 * MIB, 1, 1, &[1024, 1024])); // 4 MiB tensor
        let no_alloc = big.verdict(
            &DeviceLimits {
                vram_mb: 8192,
                ram_mb: 8192,
                max_alloc_mb: 2,
                ..Default::default()
            },
            64,
        );
        assert!(!no_alloc.eligible);
        assert!(no_alloc.reasons[0].contains("max single allocation"));
    }

    #[test]
    fn oom_error_class_is_out_of_memory() {
        assert_eq!(
            oom_error_class(),
            daemon_swarm_run::protocol::ErrorClass::OutOfMemory
        );
    }

    // ---- UMA / unified-memory autotune (the Merge-2 fix) ----

    /// A representative fp32 resource model for an N-parameter LLaMA (spec §5.1 fp32 storage): the
    /// fixed device term is params + fp32 master + fp32 grad + AdamW m/v persistents = 20·N bytes;
    /// host RAM ≈ masters + round base = 8·N; the largest single tensor is the tied embedding
    /// (`vocab·d_model` fp32). `act_per_mb` is a representative per-micro-batch activation cost.
    fn llama_model(n_params: u64, vocab: u64, d_model: u64, act_per_mb: u64) -> Autotune {
        Autotune {
            fixed_vram_bytes: 20 * n_params, // 4N storage + 4N master + 4N grad + 8N Adam m/v
            act_bytes_per_mb: act_per_mb,
            host_ram_bytes: 8 * n_params,
            payload_bytes: 4 * n_params / 64, // sparse_loco 1/64 density, illustrative
            max_tensor_bytes: vocab * d_model * 4,
        }
    }

    /// The Strix Halo (this machine) unified-memory limits: dedicated VRAM 4096 MiB, GTT/shared
    /// 120000 MiB, host RAM ~124419 MiB, per-buffer clamp 2047 MiB (wgpu-hal's i32::MAX on
    /// Linux/Mesa), `unified = true`.
    fn strix_halo() -> DeviceLimits {
        DeviceLimits {
            vram_mb: 4096,
            ram_mb: 124_419,
            max_alloc_mb: 2047,
            shared_mb: 120_000,
            unified: true,
        }
    }

    /// `unified_verdict_admits_160m_and_1_2b`: on a unified device the joint-pool budget
    /// (`min(vram + 90%·gtt, ram)`) admits both the 160M and 1.2B presets that the old
    /// VRAM-vs-`max_buffer_size` (2047 MiB) check wrongly rejected — while the per-buffer ceiling
    /// still bites an oversized single tensor.
    #[test]
    fn unified_verdict_admits_160m_and_1_2b() {
        let limits = strix_halo();
        // Effective budget = 4096 + 0.9·120000 = 112096 MiB; joint pool = min(112096, 124419).
        let effective = 4096 + 120_000 * 9 / 10;
        assert_eq!(effective, 112_096);

        // 160M preset (M1: N=151,862,784, vocab 50257, d_model 768). fixed ≈ 2897 MiB, host ≈ 1159.
        let m160 = llama_model(151_862_784, 50_257, 768, 128 * MIB);
        let v = m160.verdict(&limits, DEFAULT_MAX_MICROBATCH);
        assert!(v.eligible, "160M must be eligible on unified DRAM: {v:?}");
        assert!(v.micro_batch >= 1);
        // The device footprint alone (≈2897 fixed) exceeds the 2047 MiB per-buffer proxy the old
        // path used as the whole VRAM budget — proof the joint pool is what admits it.
        assert!((m160.fixed_vram_bytes / MIB) > limits.max_alloc_mb);

        // 1.2B preset (representative: N=1.2e9, vocab 50257, d_model 2048). fixed ≈ 22888 MiB.
        let b1_2 = llama_model(1_200_000_000, 50_257, 2048, 512 * MIB);
        let v = b1_2.verdict(&limits, DEFAULT_MAX_MICROBATCH);
        assert!(v.eligible, "1.2B must be eligible on unified DRAM: {v:?}");
        assert!(v.micro_batch >= 1);

        // The per-buffer gate is untouched: a single tensor > 2047 MiB is rejected even on unified.
        let oversized = Autotune {
            max_tensor_bytes: 2100 * MIB,
            ..llama_model(151_862_784, 50_257, 768, 128 * MIB)
        };
        let v = oversized.verdict(&limits, DEFAULT_MAX_MICROBATCH);
        assert!(!v.eligible);
        assert!(v.reasons[0].contains("max single allocation"));
    }

    /// `unified_bug_repro_and_fix`: the exact Merge-2 blocker. With the pre-fix limits (VRAM = the
    /// 2047 MiB `max_buffer_size` proxy, non-unified, no shared pool) the 160M preset is rejected
    /// ("insufficient VRAM"); flipping to the real unified limits makes it eligible. Same model,
    /// only the device-limits interpretation changed.
    #[test]
    fn unified_bug_repro_and_fix() {
        let m160 = llama_model(151_862_784, 50_257, 768, 128 * MIB);

        // Pre-fix: the old code fed `vram_mb = max_alloc_mb = 2047`, non-unified, shared 0.
        let clamped = DeviceLimits {
            vram_mb: 2047,
            ram_mb: 124_419,
            max_alloc_mb: 2047,
            shared_mb: 0,
            unified: false,
        };
        let before = m160.verdict(&clamped, DEFAULT_MAX_MICROBATCH);
        assert!(!before.eligible, "reproduces the blocker: {before:?}");
        assert!(before.reasons[0].contains("VRAM"));

        // Post-fix: real unified limits (sysfs VRAM 4096 + GTT 120000, unified).
        let after = m160.verdict(&strix_halo(), DEFAULT_MAX_MICROBATCH);
        assert!(after.eligible, "the UMA fix admits 160M: {after:?}");
    }

    /// `discrete_path_unchanged`: a classic discrete GPU (`unified = false`, `shared_mb = 0`) keeps
    /// the independent VRAM + RAM checks exactly as before — a big card admits the model, a
    /// genuinely VRAM-starved card still rejects it (the clamp is only wrong for unified memory).
    #[test]
    fn discrete_path_unchanged() {
        let m160 = llama_model(151_862_784, 50_257, 768, 128 * MIB);

        // 24 GB discrete card, no shared pool → eligible via the discrete VRAM loop.
        let big = DeviceLimits {
            vram_mb: 24_000,
            ram_mb: 64_000,
            max_alloc_mb: 4096,
            shared_mb: 0,
            unified: false,
        };
        assert!(m160.verdict(&big, DEFAULT_MAX_MICROBATCH).eligible);

        // A genuinely 2 GB discrete card with ample RAM → still (correctly) ineligible on VRAM.
        let tiny = DeviceLimits {
            vram_mb: 2047,
            ram_mb: 64_000,
            max_alloc_mb: 2047,
            shared_mb: 0,
            unified: false,
        };
        let v = m160.verdict(&tiny, DEFAULT_MAX_MICROBATCH);
        assert!(
            !v.eligible,
            "a real 2 GB discrete card cannot fit 160M: {v:?}"
        );
        assert!(v.reasons[0].contains("VRAM"));
    }

    /// `parse_amdgpu_mem_mb_fixtures`: the sysfs byte-count parser (amdgpu `mem_info_*_total`).
    #[test]
    fn parse_amdgpu_mem_mb_fixtures() {
        // This machine's real values: 4 GiB VRAM, 120000 MiB GTT.
        assert_eq!(parse_amdgpu_mem_mb("4294967296\n"), Some(4096));
        assert_eq!(parse_amdgpu_mem_mb("125829120000\n"), Some(120_000));
        assert_eq!(parse_amdgpu_mem_mb("  4294967296  "), Some(4096));
        // Non-numeric / empty → None (caller falls back to another source).
        assert_eq!(parse_amdgpu_mem_mb(""), None);
        assert_eq!(parse_amdgpu_mem_mb("N/A"), None);
    }

    // ---- Windows DXGI/D3D12 probe mapping (swarm-windows-vram-design.md §2) ----

    const GIB: u64 = 1024 * MIB;

    /// `windows_discrete_maps_dedicated_vram_shared_zero`: a discrete card (RTX-5090-shaped: 32 GiB
    /// dedicated, NON_LOCAL budget ≈ ½ RAM) maps to `vram_mb = DedicatedVideoMemory`, `shared_mb = 0`
    /// (NON_LOCAL spill contributes 0 by default), `unified = false`. The NON_LOCAL budget is carried
    /// on the raw struct for telemetry but never enters the verdict.
    #[test]
    fn windows_discrete_maps_dedicated_vram_shared_zero() {
        let adapter = DxgiAdapterMemory {
            dedicated_video: 32 * GIB,
            dedicated_system: 0,
            shared_system: 32 * GIB, // ≈ ½ of 64 GiB RAM
            is_software: false,
            uma: false,
            cache_coherent_uma: false,
            budget_local: 30 * GIB, // ≈ 0.9 × VRAM (what Task Manager shows)
            budget_non_local: 30 * GIB, // ≈ ½ RAM (telemetry only)
        };
        let limits =
            windows_device_limits(&adapter, 64 * 1024, DX12_MAX_BUFFER_MB).expect("not WARP");
        assert_eq!(limits.vram_mb, 32 * 1024);
        assert_eq!(limits.shared_mb, 0, "discrete NON_LOCAL contributes 0");
        assert!(!limits.unified);
        assert_eq!(limits.max_alloc_mb, DX12_MAX_BUFFER_MB); // 2047 — the DX12 constant
                                                             // Effective budget on the discrete path = vram only (shared 0).
        let effective = limits.vram_mb + limits.shared_mb * 9 / 10;
        assert_eq!(effective, 32 * 1024);
    }

    /// `windows_uma_uses_local_budget`: an integrated/UMA adapter (small dedicated carve-out, large
    /// LOCAL budget) maps to `unified = true`, `shared_mb = min(SharedSystemMemory, LOCAL.Budget)`.
    #[test]
    fn windows_uma_uses_local_budget() {
        let adapter = DxgiAdapterMemory {
            dedicated_video: 512 * MIB, // iGPU carve-out
            dedicated_system: 0,
            shared_system: 16 * GIB, // static ceiling (~½ of 32 GiB)
            is_software: false,
            uma: true,
            cache_coherent_uma: true,
            budget_local: 12 * GIB, // live shared-pool grant (< static ceiling)
            budget_non_local: 0,    // UMA => ~0
        };
        let limits =
            windows_device_limits(&adapter, 32 * 1024, DX12_MAX_BUFFER_MB).expect("not WARP");
        assert!(limits.unified);
        assert_eq!(limits.vram_mb, 512);
        // min(16 GiB ceiling, 12 GiB live budget) = 12 GiB.
        assert_eq!(limits.shared_mb, 12 * 1024);
    }

    /// `windows_variable_graphics_memory_clamped_to_ram`: the AMD "Variable Graphics Memory" trap —
    /// a Strix-Halo-on-Windows APU can present tens of GB as `DedicatedVideoMemory`. The mapper keeps
    /// the configured allocation in `vram_mb` (never conflated with RAM) and the verdict clamps the
    /// joint pool to physical `ram_mb`, so an inflated VRAM figure cannot overstate capacity.
    #[test]
    fn windows_variable_graphics_memory_clamped_to_ram() {
        let adapter = DxgiAdapterMemory {
            dedicated_video: 48 * GIB, // configured VGM allocation (huge)
            dedicated_system: 0,
            shared_system: 60 * GIB,
            is_software: false,
            uma: true,
            cache_coherent_uma: true,
            budget_local: 56 * GIB,
            budget_non_local: 0,
        };
        let ram_mb = 128 * 1024; // 128 GiB physical
        let limits = windows_device_limits(&adapter, ram_mb, DX12_MAX_BUFFER_MB).expect("not WARP");
        assert_eq!(limits.vram_mb, 48 * 1024, "configured allocation, not RAM");
        assert!(limits.unified);
        // `vram_mb` holds only the configured allocation — it is NOT summed with `shared_mb` into a
        // fake capacity (the design's cardinal VGM trap rule).
        assert!(limits.vram_mb < limits.vram_mb + limits.shared_mb);
        // The joint pool = min(vram + 90%·shared, ram) is bounded by physical `ram_mb`, so no VGM
        // configuration can admit a model whose footprint exceeds physical RAM.
        let over_ram = Autotune {
            fixed_vram_bytes: 200 * GIB, // far beyond 128 GiB RAM
            act_bytes_per_mb: MIB,
            host_ram_bytes: MIB,
            payload_bytes: 0,
            max_tensor_bytes: MIB,
        };
        let v = over_ram.verdict(&limits, DEFAULT_MAX_MICROBATCH);
        assert!(!v.eligible, "VGM must not admit a >RAM model: {v:?}");
        // A model that fits physical RAM IS admitted (the configured VGM allocation is usable).
        let fits = llama_model(1_200_000_000, 50_257, 2048, 256 * MIB); // ~22 GiB fixed
        assert!(
            fits.verdict(&limits, DEFAULT_MAX_MICROBATCH).eligible,
            "a RAM-fitting model must be eligible on VGM: {:?}",
            fits.verdict(&limits, DEFAULT_MAX_MICROBATCH)
        );
    }

    /// `windows_warp_skipped`: a software (WARP) adapter maps to `None` so enumeration skips it.
    #[test]
    fn windows_warp_skipped() {
        let warp = DxgiAdapterMemory {
            dedicated_video: 0,
            is_software: true,
            ..Default::default()
        };
        assert!(windows_device_limits(&warp, 16 * 1024, DX12_MAX_BUFFER_MB).is_none());
    }

    // ---- macOS Metal probe mapping (swarm-macos-uma-findings.md §4) ----

    /// `macos_m1_working_set_and_joint_pool`: the measured M1-mini numbers (8 GiB) map to
    /// `vram_mb = recommendedMaxWorkingSetSize` (⅔ RAM), `max_alloc_mb = maxBufferLength` (½ RAM),
    /// `shared_mb = ram_mb`, `unified = true`; the joint pool then admits 160M and rejects 1.2B.
    #[test]
    fn macos_m1_working_set_and_joint_pool() {
        let metal = MetalAdapterMemory {
            recommended_working_set: 5_726_633_984, // 5461 MiB (⅔ of 8 GiB, measured)
            max_buffer_length: 4 * GIB,             // 4096 MiB (½ of 8 GiB, measured)
            phys_ram: 8 * GIB,
            has_unified: true,
        };
        let limits = macos_device_limits(&metal);
        assert_eq!(limits.vram_mb, 5461);
        assert_eq!(limits.max_alloc_mb, 4096);
        assert_eq!(limits.ram_mb, 8192);
        assert_eq!(limits.shared_mb, 8192);
        assert!(limits.unified);

        // 160M ELIGIBLE, 1.2B INELIGIBLE on this 8 GiB M1 (findings §3).
        let m160 = llama_model(151_862_784, 50_257, 768, 8 * MIB);
        assert!(
            m160.verdict(&limits, DEFAULT_MAX_MICROBATCH).eligible,
            "160M must fit an 8 GiB M1"
        );
        let b1_2 = llama_model(1_200_000_000, 50_257, 2048, 512 * MIB);
        assert!(
            !b1_2.verdict(&limits, DEFAULT_MAX_MICROBATCH).eligible,
            "1.2B cannot fit an 8 GiB M1"
        );
    }
}
