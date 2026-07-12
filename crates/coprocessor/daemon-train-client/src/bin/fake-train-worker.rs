// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Test-fixture worker binary; its fs (a scripted spawn-counter file) is test-only and
// daemon-controlled. Raw fs allowed file-wide (mirrors `fake-infer-worker`).
#![allow(clippy::disallowed_methods)]

//! A scripted fake `daemon-train` worker for [`TrainSupervisor`] integration tests.
//!
//! It speaks the real [`daemon_swarm_run::protocol`] over the same length-framed stdio cut as the
//! production worker, but plays a scenario selected by `DAEMON_FAKE_SCENARIO` instead of running an
//! engine, optionally varying behavior by spawn index (a counter persisted in `DAEMON_FAKE_STATE`)
//! so a test can assert "crash once, then succeed on the respawn".
//!
//! Scenarios: `ready` (default) | `exit-on-start` | `crash-once` | `ineligible`.
//!
//! `ineligible` answers `AssessRun` with a not-eligible verdict + reasons (the RUN-10 staged-assess
//! rejection path). The real meta-mode assess (static import scan + host meta pass) lives in the
//! `daemon-train-worker` binary; this fixture stays scripted so the supervision tests
//! (respawn / meltdown / crash-once) do not need the wasm engine.

use daemon_provision::{CutChannel, CutWriter};
use daemon_swarm_run::protocol::{self, Command, Eligibility, Event, Hardware, WorkerCapabilities};

#[tokio::main]
async fn main() {
    let scenario = std::env::var("DAEMON_FAKE_SCENARIO").unwrap_or_else(|_| "ready".to_string());
    let spawn_index = bump_spawn_counter();

    if scenario == "exit-on-start" {
        std::process::exit(1);
    }

    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    // Every healthy spawn announces readiness first.
    send(
        &writer,
        &Event::Ready {
            capabilities: capabilities(),
        },
    )
    .await;

    // `crash-once` misbehaves on the first spawn (index 0) and behaves on the respawn (index >= 1).
    let misbehave = scenario == "crash-once" && spawn_index == 0;

    while let Some(bytes) = reader.recv().await {
        let cmd: Command = match protocol::decode(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("fake-train-worker: undecodable command: {e}");
                continue;
            }
        };
        match cmd {
            Command::Probe => {
                if misbehave {
                    std::process::exit(1);
                }
                send(&writer, &Event::Probed(hardware())).await;
            }
            Command::AssessRun { .. } => {
                if misbehave {
                    std::process::exit(1);
                }
                let assessed = if scenario == "ineligible" {
                    Eligibility {
                        eligible: false,
                        reasons: vec!["fake: vram below floor".into()],
                        headroom: vec![("vram_mb".into(), -2048)],
                    }
                } else {
                    Eligibility {
                        eligible: true,
                        reasons: vec!["fake: fits".into()],
                        headroom: vec![("vram_mb".into(), 4096)],
                    }
                };
                send(&writer, &Event::Assessed(assessed)).await;
            }
            Command::JoinRun { run_id, .. } => {
                if misbehave {
                    std::process::exit(1);
                }
                send(
                    &writer,
                    &Event::RunPhase {
                        run_id,
                        phase: "warmup".into(),
                        epoch: 0,
                        round: 0,
                    },
                )
                .await;
            }
            Command::Throttle { .. } | Command::Leave { .. } => {
                // One-way commands: no reply.
            }
            Command::Ping => send(&writer, &Event::Pong).await,
            Command::Shutdown => break,
        }
    }
}

fn capabilities() -> WorkerCapabilities {
    WorkerCapabilities {
        abi_version: 1,
        ops: vec!["matmul@1".into()],
        payload_stores: vec!["r2".into()],
    }
}

fn hardware() -> Hardware {
    Hardware {
        gpus: 1,
        vram_mb: 24_000,
        ram_mb: 64_000,
        backend_lanes: vec!["cpu".into()],
        capabilities: capabilities(),
        up_kbps: 50_000,
        down_kbps: 200_000,
        disk_free_mb: 500_000,
        throughput_class: "c3".into(),
    }
}

async fn send(writer: &CutWriter, event: &Event) {
    let bytes = protocol::encode(event).expect("encode event");
    let _ = writer.send(&bytes).await;
}

/// Read-increment the spawn counter in `DAEMON_FAKE_STATE` (if set), returning this spawn's index.
fn bump_spawn_counter() -> u64 {
    let Ok(path) = std::env::var("DAEMON_FAKE_STATE") else {
        return 0;
    };
    let current = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let _ = std::fs::write(&path, (current + 1).to_string());
    current
}
