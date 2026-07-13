// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// The worker reads its module path from an env var and the module bytes from disk (developer /
// node-controlled inputs, mirroring `fake-train-worker`); the fs/env hardening bans target the
// shipped node process, not this isolated worker binary. Allowed file-wide (crate-level, so the
// `transport`/`backend` submodules inherit it too).
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
//! - `AssessRun{envelope}` → the peer-side re-validation (spec §6.5). The `envelope` bytes are the
//!   canonical [`daemon_swarm_proto::SignedEnvelope`] wire form of the run's `FrozenEnvelope`: the
//!   worker **verifies** it, extracts the `[experiment.config]`, and **resolves the module** from the
//!   envelope's artifact map via [`daemon_swarm_net::ArtifactResolver`] (`file://`, blake3-verified).
//!   `DAEMON_TRAIN_MODULE`, if set, overrides the artifact resolution (dev / node-controlled). It then
//!   runs the static import scan vs the host vocabulary + a host meta-mode pass → `Assessed`, caching
//!   the config + module bytes for the subsequent `JoinRun`. A raw-config-CBOR envelope (no signature
//!   wrapper) is still accepted as a legacy path (module from `DAEMON_TRAIN_MODULE`).
//! - `JoinRun` → construct a `WasmBackend`, emit `RunPhase{train}`, self-drive one round
//!   (train × H → make_update → ingest) and stream `Metric`/`RoundOutcome`.
//! - `Throttle{paused}` → `WasmBackend::pause`/`resume` (preemption-as-churn, §10.5).
//! - `Leave`/`Shutdown`/`Ping` → as the protocol requires.
//!
//! A trapping module surfaces as `Event::Error{class: Module, …}` — the worker is never harmed.
//!
//! ## Module layout (Wave-0 scaffold split — see swarm-p1-ledger.md)
//!
//! This binary is split into two sides so the Wave-3 A/B lanes do not collide on one file:
//! - [`backend`] — the `WasmBackend` construction / assess / probe side (**G2** owns it).
//! - [`transport`] — the `JoinRun` / coordinator-attach side; today the self-driven round loop,
//!   which **B3** replaces with a live coordinator connection (`JoinRun.coordinator`) in Wave 3.
//!
//! `main` is the thin command dispatch loop plus the shared `send`/`worker_error` helpers and the
//! representative micro-batch shape ([`SEQS`]/[`SEQ`]) both sides use.

mod backend;
mod transport;

use daemon_provision::{CutChannel, CutWriter};
use daemon_swarm_run::protocol::{self, Command, ErrorClass, Event};
use daemon_train::WasmBackend;

/// A representative meta/self-drive micro-batch shape (sequences × tokens-per-sequence). All-zero
/// token ids are valid for any vocabulary (id 0 always exists), so the worker stays experiment
/// agnostic (it drives `da_*`, it does not know the model's vocab). Shared by `backend`'s meta pass
/// and `transport`'s self-driven round.
pub(crate) const SEQS: u32 = 2;
pub(crate) const SEQ: u32 = 8;

#[tokio::main]
async fn main() {
    // Consent-gated crash reporting (component = train-worker). Armed as the first action: the
    // minidump monitor re-exec's this binary with a `--crash-reporter-server` arg, and this init
    // runs the monitor server (then exits) in that copy before it touches the stdio cut. A no-op
    // unless the spawning node injected a DSN + `DAEMON_CRASH_CONSENT=1`.
    let _crash = daemon_telemetry::init_crash_reporting("train-worker");
    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    send(
        &writer,
        &Event::Ready {
            capabilities: backend::host_capabilities(),
        },
    )
    .await;

    // Cached across commands: the assessed run (config + module bytes) + the live joined backend +
    // the micro-batch the last `AssessRun` autotune chose (G2's `Eligibility.headroom["micro_batch"]`),
    // threaded into `JoinRun` so the worker consumes the verdict in-process (B3 lifecycle glue).
    let mut run: Option<backend::ResolvedRun> = None;
    let mut live_backend: Option<WasmBackend> = None;
    let mut assessed_micro_batch: u32 = SEQS;

    while let Some(bytes) = reader.recv().await {
        let cmd: Command = match protocol::decode(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("daemon-train-worker: undecodable command: {e}");
                continue;
            }
        };
        match cmd {
            Command::Probe => send(&writer, &Event::Probed(backend::hardware())).await,
            Command::AssessRun { envelope } => match backend::resolve_run(&envelope).await {
                Ok(resolved) => match backend::assess(&resolved.module, &resolved.config) {
                    Ok(elig) => {
                        // Consume the autotune micro-batch (G2 rides it in `headroom["micro_batch"]`)
                        // so `JoinRun` drives / OOM-probes from the node-computed verdict (§10.5).
                        if let Some((_, mb)) =
                            elig.headroom.iter().find(|(k, _)| k == "micro_batch")
                        {
                            assessed_micro_batch = (*mb).max(1) as u32;
                        }
                        run = Some(resolved);
                        send(&writer, &Event::Assessed(elig)).await;
                    }
                    Err(detail) => send(&writer, &worker_error(&detail)).await,
                },
                Err(detail) => send(&writer, &worker_error(&detail)).await,
            },
            Command::JoinRun { run_id, .. } => {
                let Some(resolved) = run.as_ref() else {
                    send(
                        &writer,
                        &worker_error("JoinRun before AssessRun: no resolved run"),
                    )
                    .await;
                    continue;
                };
                match transport::join_and_run_round(
                    &resolved.module,
                    &resolved.config,
                    &run_id,
                    assessed_micro_batch,
                    &writer,
                )
                .await
                {
                    Ok(b) => live_backend = Some(b),
                    Err(detail) => send(&writer, &worker_error(&detail)).await,
                }
            }
            Command::Throttle { paused, .. } => {
                if let Some(b) = live_backend.as_mut() {
                    let r = if paused { b.pause() } else { b.resume() };
                    if let Err(e) = r {
                        send(&writer, &worker_error(&e.to_string())).await;
                    }
                }
            }
            Command::Leave { .. } => live_backend = None,
            Command::Ping => send(&writer, &Event::Pong).await,
            Command::Shutdown => break,
        }
    }
}

/// A module trap / lifecycle failure surfaces as the `Module` error class (worker unharmed, §13).
pub(crate) fn worker_error(detail: &str) -> Event {
    Event::Error {
        class: ErrorClass::Module,
        detail: detail.to_string(),
    }
}

/// Encode and send an [`Event`] over the stdio cut (shared by `main` and `transport`).
pub(crate) async fn send(writer: &CutWriter, event: &Event) {
    match protocol::encode(event) {
        Ok(bytes) => {
            let _ = writer.send(&bytes).await;
        }
        Err(e) => eprintln!("daemon-train-worker: encode event: {e}"),
    }
}
