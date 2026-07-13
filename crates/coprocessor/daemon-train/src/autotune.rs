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
}
