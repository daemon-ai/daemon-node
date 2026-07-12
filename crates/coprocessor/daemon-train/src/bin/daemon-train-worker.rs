// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// The worker reads its module path from an env var and the module bytes from disk (developer /
// node-controlled inputs, mirroring `fake-train-worker`); the fs/env hardening bans target the
// shipped node process, not this isolated worker binary. Allowed file-wide.
#![allow(clippy::disallowed_methods)]
#![forbid(unsafe_code)]

//! The `daemon-train-worker` binary — the child side of the frozen worker protocol (§10.2).
//!
//! Speaks [`daemon_swarm_run::protocol`] `Command`/`Event` frames over the length-framed
//! [`daemon_provision::CutChannel`] stdio cut (exactly like `fake-train-worker`, and consumed by
//! `daemon-train-client::TrainSupervisor`), but drives the real [`daemon_train::WasmBackend`] host
//! runtime instead of a script:
//!
//! - `Probe` → a real host capability report (`tabi@1`, all 66 host ops; GPU absent = CPU-only).
//! - `AssessRun{envelope}` → the peer-side re-validation (spec §6.5): a static import scan of the
//!   module bytes vs the host vocabulary, then a host meta-mode pass → `Assessed(Eligibility)`. The
//!   envelope bytes are the `[experiment.config]` CBOR (real `FrozenEnvelope`/artifact resolution is
//!   the Merge-3 seam); the config is cached for the subsequent `JoinRun`.
//! - `JoinRun` → construct a `WasmBackend`, emit `RunPhase{train}`, self-drive one round
//!   (train × H → make_update → ingest) and stream `Metric`/`RoundOutcome`.
//! - `Throttle{paused}` → `WasmBackend::pause`/`resume` (preemption-as-churn, §10.5).
//! - `Leave`/`Shutdown`/`Ping` → as the protocol requires.
//!
//! A trapping module surfaces as `Event::Error{class: Module, …}` — the worker is never harmed.
//!
//! The experiment `.wasm` module is located via `DAEMON_TRAIN_MODULE` (an absolute path).

use std::collections::BTreeSet;

use daemon_provision::{CutChannel, CutWriter};
use daemon_swarm_proto::{blake3_hash, PeerId};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StepCtx, TrainerBackend};
use daemon_swarm_run::protocol::{
    self, Command, Eligibility, ErrorClass, Event, Hardware, WorkerCapabilities,
};
use daemon_train::phase::PHASE_TABLE;
use daemon_train::{EngineConfig, WasmBackend, WasmBackendConfig, WasmBackendError};

/// A representative meta/self-drive micro-batch shape (sequences × tokens-per-sequence). All-zero
/// token ids are valid for any vocabulary (id 0 always exists), so the worker stays experiment
/// agnostic (it drives `da_*`, it does not know the model's vocab).
const SEQS: u32 = 2;
const SEQ: u32 = 8;

/// A reserved self-peer id for the MVP self-driven round's committed set (not a real node identity).
const SELF_PEER: PeerId = PeerId([0xA1; 32]);

#[tokio::main]
async fn main() {
    let module = match load_module_bytes() {
        Ok(bytes) => bytes,
        Err(detail) => {
            // No channel yet on a hard config error — report on stderr and exit non-zero so the
            // supervisor sees an unhealthy spawn.
            eprintln!("daemon-train-worker: {detail}");
            std::process::exit(1);
        }
    };

    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    send(
        &writer,
        &Event::Ready {
            capabilities: host_capabilities(),
        },
    )
    .await;

    // Cached across commands: the last assessed `[experiment.config]` + the live joined backend.
    let mut config: Option<Vec<u8>> = None;
    let mut backend: Option<WasmBackend> = None;

    while let Some(bytes) = reader.recv().await {
        let cmd: Command = match protocol::decode(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("daemon-train-worker: undecodable command: {e}");
                continue;
            }
        };
        match cmd {
            Command::Probe => send(&writer, &Event::Probed(hardware())).await,
            Command::AssessRun { envelope } => {
                config = Some(envelope.clone());
                match assess(&module, &envelope) {
                    Ok(elig) => send(&writer, &Event::Assessed(elig)).await,
                    Err(detail) => send(&writer, &worker_error(&detail)).await,
                }
            }
            Command::JoinRun { run_id, .. } => {
                let Some(cfg) = config.clone() else {
                    send(
                        &writer,
                        &worker_error("JoinRun before AssessRun: no experiment config"),
                    )
                    .await;
                    continue;
                };
                match join_and_run_round(&module, &cfg, &run_id, &writer).await {
                    Ok(b) => backend = Some(b),
                    Err(detail) => send(&writer, &worker_error(&detail)).await,
                }
            }
            Command::Throttle { paused, .. } => {
                if let Some(b) = backend.as_mut() {
                    let r = if paused { b.pause() } else { b.resume() };
                    if let Err(e) = r {
                        send(&writer, &worker_error(&e.to_string())).await;
                    }
                }
            }
            Command::Leave { .. } => backend = None,
            Command::Ping => send(&writer, &Event::Pong).await,
            Command::Shutdown => break,
        }
    }
}

/// The `.wasm` module bytes from `DAEMON_TRAIN_MODULE`.
fn load_module_bytes() -> Result<Vec<u8>, String> {
    let path = std::env::var("DAEMON_TRAIN_MODULE")
        .map_err(|_| "DAEMON_TRAIN_MODULE is not set (path to the experiment .wasm)".to_string())?;
    std::fs::read(&path).map_err(|e| format!("reading module {path}: {e}"))
}

/// The host `tabi@1` vocabulary (name-for-name with the phase table / SDK `TABI_IMPORTS`, all 66).
fn host_ops() -> Vec<String> {
    PHASE_TABLE.iter().map(|(n, _)| (*n).to_string()).collect()
}

fn host_capabilities() -> WorkerCapabilities {
    WorkerCapabilities {
        abi_version: daemon_train::TENSOR_ABI_MAJOR as u16,
        ops: host_ops(),
        payload_stores: Vec::new(),
    }
}

/// A CPU-only host report (no GPU: this build carries no GPU backend lanes, §10.1).
fn hardware() -> Hardware {
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
fn assess(module: &[u8], config: &[u8]) -> Result<Eligibility, String> {
    let worker =
        daemon_train::Worker::new(EngineConfig::default()).map_err(|e| format!("engine: {e}"))?;
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

/// Construct the backend, emit `RunPhase{train}` (the supervisor's `join` resolves here), then
/// self-drive one round and stream `Metric`/`RoundOutcome`. Returns the live backend (kept for
/// `Throttle`). The round loop is self-driven for the MVP — connecting to `JoinRun.coordinator` is a
/// Merge-3 decision (see the E3 ledger).
async fn join_and_run_round(
    module: &[u8],
    config: &[u8],
    run_id: &str,
    writer: &CutWriter,
) -> Result<WasmBackend, String> {
    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: module.to_vec(),
        engine: EngineConfig::default(),
    })
    .map_err(err_detail)?;
    backend.build(config).map_err(err_detail)?;
    let steps = backend.steps_per_round().map_err(err_detail)?;

    send(
        writer,
        &Event::RunPhase {
            run_id: run_id.to_string(),
            phase: "train".to_string(),
            epoch: 0,
            round: 0,
        },
    )
    .await;

    let mut last_loss = f32::NAN;
    for step in 0..steps {
        let stats = backend
            .train_step(
                &BatchRef {
                    tokens: vec![0u32; (SEQS * SEQ) as usize],
                    seq_len: SEQ,
                },
                StepCtx {
                    inner_step: step,
                    mb_index: 0,
                    mb_count: 1,
                    step_seqs: SEQS,
                },
            )
            .map_err(err_detail)?;
        last_loss = stats.loss;
        backend.inner_update(step).map_err(err_detail)?;
    }

    let payload = backend.make_update(0).map_err(err_detail)?;
    let digest = backend
        .ingest(
            0,
            &[StagedPayload {
                peer: SELF_PEER,
                hash: blake3_hash(&payload),
                bytes: payload,
            }],
        )
        .map_err(err_detail)?;

    send(
        writer,
        &Event::Metric {
            name: "loss".to_string(),
            value: f64::from(last_loss),
        },
    )
    .await;
    send(
        writer,
        &Event::RoundOutcome {
            round: 0,
            committed: 1,
            ingested: 1,
            stalled: false,
            digest: *digest.as_bytes(),
        },
    )
    .await;

    Ok(backend)
}

/// Render a backend error for an `Event::Error` detail.
fn err_detail(e: WasmBackendError) -> String {
    e.to_string()
}

/// A module trap / lifecycle failure surfaces as the `Module` error class (worker unharmed, §13).
fn worker_error(detail: &str) -> Event {
    Event::Error {
        class: ErrorClass::Module,
        detail: detail.to_string(),
    }
}

async fn send(writer: &CutWriter, event: &Event) {
    match protocol::encode(event) {
        Ok(bytes) => {
            let _ = writer.send(&bytes).await;
        }
        Err(e) => eprintln!("daemon-train-worker: encode event: {e}"),
    }
}
