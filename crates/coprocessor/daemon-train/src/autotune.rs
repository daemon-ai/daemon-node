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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Effective total VRAM in MiB (governor policy cap / node effective resources — NOT
    /// wgpu-queryable; see the module docs).
    pub vram_mb: u64,
    /// Effective host RAM in MiB.
    pub ram_mb: u64,
    /// Largest single GPU allocation in MiB (wgpu `max_buffer_size`); `0` = unknown / unbounded
    /// (the CPU / ndarray lanes, or when no device was probed).
    pub max_alloc_mb: u64,
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
    /// analytical form). Floor of 1 → ineligible if a single sequence does not fit VRAM.
    #[must_use]
    pub fn verdict(&self, limits: &DeviceLimits, max_microbatch: u32) -> AutotuneVerdict {
        let vram_budget = limits.vram_mb.saturating_mul(MIB);
        let ram_budget = limits.ram_mb.saturating_mul(MIB);
        let max_alloc = if limits.max_alloc_mb == 0 {
            u64::MAX
        } else {
            limits.max_alloc_mb.saturating_mul(MIB)
        };

        let ineligible = |reason: String| AutotuneVerdict {
            eligible: false,
            micro_batch: 0,
            vram_mb_estimate: self.fixed_vram_bytes.div_ceil(MIB).max(1),
            ram_mb_estimate: self.host_ram_bytes.div_ceil(MIB).max(1),
            payload_bytes_estimate: self.payload_bytes,
            oom_retries: 0,
            reasons: vec![reason],
        };

        // A single param master must fit one GPU buffer (the wgpu-queryable hard ceiling).
        if self.max_tensor_bytes > max_alloc {
            return ineligible(format!(
                "largest tensor {} MiB exceeds max single allocation {} MiB",
                self.max_tensor_bytes.div_ceil(MIB),
                limits.max_alloc_mb
            ));
        }
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
            if need <= vram_budget {
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
                    limits.vram_mb
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
    /// also used as the VRAM proxy.
    pub max_alloc_mb: u64,
    /// The adapter name (`get_info().name`, e.g. "AMD Radeon … (RADV …)").
    pub adapter: String,
    /// The graphics backend (`get_info().backend`, e.g. "Vulkan").
    pub backend: String,
}

/// Probe the default wgpu device for its adapter info + limits (feature `wgpu`). Returns `None`
/// (never panics) when no adapter can be brought up. Uses cubecl's `init_setup` under
/// `catch_unwind`; reads only wgpu-queryable fields (`get_info`, `limits`).
///
/// **Memoized process-wide**: cubecl's `ComputeClient::init` panics if the default device's client
/// is already registered, so the setup bring-up must happen exactly once — this probe IS that
/// bring-up (subsequent burn tensor ops on the default device reuse the registered client), and
/// every later call returns the cached result. Call it (or [`crate::wgpu_adapter_available`],
/// which delegates here) before any wgpu tensor work in the process.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn probe_wgpu() -> Option<WgpuProbe> {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions, WgpuDevice};
    use std::sync::OnceLock;

    static PROBE: OnceLock<Option<WgpuProbe>> = OnceLock::new();
    PROBE
        .get_or_init(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let probe = std::panic::catch_unwind(|| {
                let setup = init_setup::<AutoGraphicsApi>(
                    &WgpuDevice::DefaultDevice,
                    RuntimeOptions::default(),
                );
                let info = setup.adapter.get_info();
                let limits = setup.adapter.limits();
                let max_alloc_mb = (limits.max_buffer_size / MIB).max(1);
                WgpuProbe {
                    gpus: 1,
                    max_alloc_mb,
                    adapter: info.name.clone(),
                    backend: format!("{:?}", info.backend),
                }
            })
            .ok();
            std::panic::set_hook(prev);
            probe
        })
        .clone()
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
}
