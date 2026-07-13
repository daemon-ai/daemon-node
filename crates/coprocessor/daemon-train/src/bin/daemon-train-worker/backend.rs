// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `WasmBackend` construction / assess / probe side of the worker (§6.5, §10.2).
//!
//! Owns the `Probe` hardware report, the `AssessRun` envelope→`(config, module)` resolution, and the
//! meta-mode eligibility pass. **G2** (Wave 2) evolves this file: real GPU `Hardware` numbers, VRAM
//! autotune / OOM probe, and the burn-wgpu backend behind `WasmBackend::assess`.

use std::collections::BTreeSet;

use daemon_swarm_net::{ArtifactRef, ArtifactResolver};
use daemon_swarm_proto::{from_canonical_slice, SignedEnvelope};
use daemon_swarm_run::protocol::{Eligibility, Hardware, WorkerCapabilities};
use daemon_train::autotune::{Autotune, DeviceLimits, DEFAULT_MAX_MICROBATCH};
use daemon_train::phase::PHASE_TABLE;
use daemon_train::{EngineConfig, Worker};

use crate::SEQ;

/// A large sentinel (in MiB) used when a resource dimension is unknown, so the autotune verdict does
/// not spuriously reject on an unprobed number (`u64::MAX / MiB`).
const UNKNOWN_BUDGET_MB: u64 = u64::MAX / (1 << 20);

/// The experiment inputs a run resolves to: the `[experiment.config]` CBOR + the module `.wasm`.
pub(crate) struct ResolvedRun {
    pub(crate) config: Vec<u8>,
    pub(crate) module: Vec<u8>,
}

/// Resolve the `AssessRun` envelope bytes into `(config, module)` (the §6.1/§6.5 seam).
///
/// The bytes are the canonical [`SignedEnvelope`] wire form: verify it, take `config_bytes()`, and
/// resolve the module from the envelope's artifact map via [`ArtifactResolver`] (`file://`,
/// blake3-verified). `DAEMON_TRAIN_MODULE` overrides the artifact fetch. If the bytes are not a
/// signed-envelope wrapper, fall back to treating them as raw `[experiment.config]` CBOR with the
/// module from `DAEMON_TRAIN_MODULE` (the legacy direct-drive path).
pub(crate) async fn resolve_run(envelope_bytes: &[u8]) -> Result<ResolvedRun, String> {
    match from_canonical_slice::<SignedEnvelope>(envelope_bytes) {
        Ok(wire) => {
            // A signed-envelope wrapper: verify it (re-derives hash + config, checks the signature).
            let frozen = wire.open().map_err(|e| format!("verify envelope: {e}"))?;
            let config = frozen.config_bytes().to_vec();
            let module = resolve_module(&frozen).await?;
            Ok(ResolvedRun { config, module })
        }
        // Not a signed-envelope wrapper: the legacy raw `[experiment.config]` CBOR path.
        Err(_) => {
            let module = module_from_env().ok_or_else(|| {
                "AssessRun envelope is neither a signed envelope nor is DAEMON_TRAIN_MODULE set"
                    .to_string()
            })??;
            Ok(ResolvedRun {
                config: envelope_bytes.to_vec(),
                module,
            })
        }
    }
}

/// Resolve the experiment module bytes for a verified envelope: `DAEMON_TRAIN_MODULE` if set
/// (override), else the envelope's `experiment.module` artifact via the `file://` resolver.
async fn resolve_module(frozen: &daemon_swarm_proto::FrozenEnvelope) -> Result<Vec<u8>, String> {
    if let Some(bytes) = module_from_env() {
        return bytes;
    }
    let envelope = frozen
        .decode()
        .map_err(|e| format!("decode envelope: {e}"))?;
    let name = &envelope.experiment.module;
    let artifact = envelope
        .artifacts
        .get(name)
        .ok_or_else(|| format!("experiment module `{name}` absent from [artifacts]"))?;
    let art = ArtifactRef::new(artifact.url.clone(), artifact.blake3);
    ArtifactResolver::new()
        .fetch(&art)
        .await
        .map_err(|e| format!("resolve module `{name}` ({}): {e}", artifact.url))
}

/// The `.wasm` module bytes from `DAEMON_TRAIN_MODULE` (the dev / node-controlled override), if set.
/// `Some(Err(..))` means the var is set but the read failed.
fn module_from_env() -> Option<Result<Vec<u8>, String>> {
    let path = std::env::var("DAEMON_TRAIN_MODULE").ok()?;
    Some(std::fs::read(&path).map_err(|e| format!("reading module {path}: {e}")))
}

/// The host `tabi@1` vocabulary (name-for-name with the phase table / SDK `TABI_IMPORTS`, all 66).
fn host_ops() -> Vec<String> {
    PHASE_TABLE.iter().map(|(n, _)| (*n).to_string()).collect()
}

pub(crate) fn host_capabilities() -> WorkerCapabilities {
    WorkerCapabilities {
        abi_version: daemon_train::TENSOR_ABI_MAJOR as u16,
        ops: host_ops(),
        payload_stores: Vec::new(),
    }
}

/// Host RAM in MiB from `/proc/meminfo` `MemTotal` (Linux, best effort). `0` if unavailable.
fn host_ram_mb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            // `MemTotal:   16384000 kB`
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())
            {
                return kb / 1024;
            }
        }
    }
    0
}

/// Read an amdgpu sysfs memory-total file for the first DRM card that exposes it, in MiB.
///
/// `file` is `mem_info_vram_total` (dedicated VRAM — the true device lower bound) or
/// `mem_info_gtt_total` (the GTT / unified spillover pool). These are plain byte-count files under
/// `/sys/class/drm/card*/device/` — a legal direct file read in the worker binary (not the node).
/// Returns `0` when no card exposes the file (non-amdgpu / non-Linux), so callers fall back.
///
/// Parsing is delegated to [`daemon_train::autotune::parse_amdgpu_mem_mb`] (unit-tested with
/// fixture strings); this wrapper only does the sysfs directory walk + read.
#[cfg(feature = "wgpu")]
fn amdgpu_sysfs_mem_mb(file: &str) -> u64 {
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    for entry in cards.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Only `cardN` device roots carry `device/mem_info_*` (skip `cardN-<connector>` outputs).
        if !(name.starts_with("card") && name[4..].bytes().all(|b| b.is_ascii_digit())) {
            continue;
        }
        let path = entry.path().join("device").join(file);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Some(mb) = daemon_train::autotune::parse_amdgpu_mem_mb(&contents) {
                if mb > 0 {
                    return mb;
                }
            }
        }
    }
    0
}

/// The host hardware + capability report (§10.2). GPU count / VRAM come from a real wgpu adapter
/// probe + sysfs when the `wgpu` feature is on; a CPU-only build reports `gpus: 0` and the CPU lane.
///
/// **VRAM source (Merge-2 UMA fix).** wgpu has no total-VRAM query and clamps `max_buffer_size` to
/// i32::MAX (2047 MiB) on Linux/Mesa — a per-buffer limit, NOT the memory budget. `vram_mb` now
/// carries the sysfs *dedicated* VRAM (`mem_info_vram_total`, a true lower bound: 4096 MiB on this
/// box) when available, falling back to the `max_buffer_size` proxy only when sysfs is absent. The
/// additive `shared_mb` carries the GTT / unified spillover pool (`mem_info_gtt_total`), which is
/// where an integrated GPU actually pages large tensors.
pub(crate) fn hardware() -> Hardware {
    let ram_mb = host_ram_mb();
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_train::autotune::probe_wgpu() {
            // Dedicated VRAM from sysfs (true lower bound); fall back to the max-alloc proxy.
            let vram_sysfs = amdgpu_sysfs_mem_mb("mem_info_vram_total");
            let vram_mb = if vram_sysfs > 0 {
                vram_sysfs
            } else {
                p.max_alloc_mb
            };
            let shared_mb = amdgpu_sysfs_mem_mb("mem_info_gtt_total");
            return Hardware {
                gpus: p.gpus,
                vram_mb,
                shared_mb,
                ram_mb,
                backend_lanes: vec!["vulkan".to_string(), "cpu".to_string()],
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: 0,
                throughput_class: "c1".to_string(),
            };
        }
    }
    Hardware {
        gpus: 0,
        vram_mb: 0,
        shared_mb: 0,
        ram_mb,
        backend_lanes: vec!["cpu".to_string()],
        capabilities: host_capabilities(),
        up_kbps: 0,
        down_kbps: 0,
        disk_free_mb: 0,
        throughput_class: "c1".to_string(),
    }
}

/// The device budget the autotune verdict is computed against (Merge-2 UMA fix).
///
/// With the `wgpu` feature + a usable adapter: `vram_mb` = sysfs dedicated VRAM (true lower bound),
/// `shared_mb` = sysfs GTT (the unified spillover pool), `max_alloc_mb` = the wgpu `max_buffer_size`
/// per-buffer ceiling, and `unified` = the adapter's device-type (IntegratedGpu/Cpu). On a unified
/// device the verdict then treats VRAM+GTT+RAM as one physical DRAM pool instead of rejecting
/// against the 2047 MiB per-buffer clamp. Without a GPU, the CPU lane runs in host RAM (no separate
/// VRAM constraint). Unknown dimensions use a large sentinel so an unprobed number never rejects.
fn device_limits() -> DeviceLimits {
    let ram_mb = {
        let r = host_ram_mb();
        if r == 0 {
            UNKNOWN_BUDGET_MB
        } else {
            r
        }
    };
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_train::autotune::probe_wgpu() {
            let vram_sysfs = amdgpu_sysfs_mem_mb("mem_info_vram_total");
            // On a unified device without sysfs VRAM, dedicated VRAM is not a meaningful cap; the
            // pool is host RAM, so budget VRAM as RAM. On a discrete device fall back to the
            // per-buffer proxy (the honest lower bound wgpu can give).
            let vram_mb = if vram_sysfs > 0 {
                vram_sysfs
            } else if p.unified {
                ram_mb
            } else {
                p.max_alloc_mb
            };
            return DeviceLimits {
                vram_mb,
                ram_mb,
                max_alloc_mb: p.max_alloc_mb,
                shared_mb: amdgpu_sysfs_mem_mb("mem_info_gtt_total"),
                unified: p.unified,
            };
        }
    }
    DeviceLimits {
        vram_mb: ram_mb,
        ram_mb,
        max_alloc_mb: 0,
        shared_mb: 0,
        unified: false,
    }
}

/// The peer-side re-validation (spec §6.5): a static import scan of the module vs the host `tabi@1`
/// vocabulary, then a host meta-mode pass over the config → an [`Eligibility`] verdict.
pub(crate) fn assess(module: &[u8], config: &[u8]) -> Result<Eligibility, String> {
    let worker = Worker::new(EngineConfig::default()).map_err(|e| format!("engine: {e}"))?;
    let vocabulary: BTreeSet<String> = host_ops().into_iter().collect();
    let imports = worker
        .module_imports(module)
        .map_err(|e| format!("module import scan: {e}"))?;
    let missing: Vec<String> = imports
        .iter()
        .filter(|name| !vocabulary.contains(name.as_str()))
        .cloned()
        .collect();

    if !missing.is_empty() {
        return Ok(Eligibility {
            eligible: false,
            reasons: vec![format!(
                "module imports ops outside host tabi@1: {}",
                missing.join(", ")
            )],
            headroom: Vec::new(),
        });
    }

    let loaded = worker
        .load_module(module)
        .map_err(|e| format!("load module: {e}"))?;
    let mut inst = worker
        .instantiate(&loaded)
        .map_err(|e| format!("instantiate: {e}"))?;
    let report = inst
        .meta(config, 1, SEQ)
        .map_err(|e| format!("meta: {e}"))?;

    // G2 VRAM autotune (§5.1 planning, ABI §8): the meta-report footprint vs the probed device
    // budget → eligibility + chosen micro-batch. The MetaReport byte footprints are
    // backend-independent (shapes/dtypes), so the CPU meta pass is authoritative for the estimates;
    // the verdict compares them against the real device numbers from `device_limits`.
    let autotune = Autotune::from_meta(&report);
    let verdict = autotune.verdict(&device_limits(), DEFAULT_MAX_MICROBATCH);

    let mib = 1i64 << 20;
    let mut reasons = vec![format!(
        "tabi@1 satisfied ({} imports); meta pass ok",
        imports.len()
    )];
    reasons.extend(verdict.reasons.iter().cloned());

    Ok(Eligibility {
        eligible: verdict.eligible,
        reasons,
        headroom: vec![
            ("micro_batch".to_string(), i64::from(verdict.micro_batch)),
            ("vram_mb".to_string(), verdict.vram_mb_estimate as i64),
            ("ram_mb".to_string(), verdict.ram_mb_estimate as i64),
            (
                "payload_bytes".to_string(),
                verdict.payload_bytes_estimate as i64,
            ),
            (
                "host_ram_mb".to_string(),
                (report.host_ram_bytes_est as i64) / mib,
            ),
            ("param_bytes".to_string(), report.param_bytes as i64),
        ],
    })
}
