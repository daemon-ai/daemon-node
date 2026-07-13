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
use daemon_train::phase::PHASE_TABLE;
use daemon_train::{EngineConfig, Worker};

use crate::SEQ;

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

/// A CPU-only host report (no GPU: this build carries no GPU backend lanes, §10.1). G2 fills in real
/// GPU count / VRAM here (currently hardcoded zeros in the CPU worker).
pub(crate) fn hardware() -> Hardware {
    Hardware {
        gpus: 0,
        vram_mb: 0,
        ram_mb: 0,
        backend_lanes: vec!["cpu".to_string()],
        capabilities: host_capabilities(),
        up_kbps: 0,
        down_kbps: 0,
        disk_free_mb: 0,
        throughput_class: "c1".to_string(),
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

    let mib = 1i64 << 20;
    Ok(Eligibility {
        eligible: true,
        reasons: vec![format!(
            "tabi@1 satisfied ({} imports); meta pass ok",
            imports.len()
        )],
        headroom: vec![
            (
                "host_ram_mb".to_string(),
                (report.host_ram_bytes_est as i64) / mib,
            ),
            ("param_bytes".to_string(), report.param_bytes as i64),
        ],
    })
}
