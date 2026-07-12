// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The training-worker wire protocol (spec §10.2) — [`Command`] (down) / [`Event`] (up) + a CBOR
//! codec.
//!
//! The node-side supervisor (`daemon-train-client`) and the `daemon-train` worker exchange these
//! frames over a length-framed stdio cut (`daemon_provision::CutChannel`, `Framing::Length`), same
//! supervision contract as `daemon-infer` (respawn with backoff, crash-loop meltdown). Each frame
//! body is CBOR; the `u32`-length prefix is handled by the channel, so this module owns only the
//! body [`encode`]/[`decode`] — the exact conventions of [`daemon_infer::protocol`].
//!
//! This is the **worker** protocol (node ↔ `daemon-train` child) — distinct from the **swarm**
//! control protocol (`daemon-swarm.cddl`, lane P). It lives in `daemon-swarm-run` (not the client)
//! so lane E's `daemon-train` worker implements the worker side against it in Wave 3 (§10.1:
//! `daemon-train` depends on `daemon-swarm-run`).

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::seam::RoundId;

/// How a peer participates on hardware primarily wanted for inference (§10.5). Mirrors the wire
/// `swarm-policy-mode` (§10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyMode {
    /// Always available for training.
    Always,
    /// Only when there is no inference activity + the user is idle.
    Idle,
    /// Within `daemon-schedule` cron windows.
    Scheduled,
    /// Manual start/stop only.
    Manual,
}

/// The participation policy for a joined run (§10.4/§10.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinPolicy {
    /// The availability mode.
    pub mode: PolicyMode,
    /// A VRAM cap in MiB (`0` = uncapped) — also tightens eligibility (§6.5).
    pub vram_cap_mb: u32,
    /// A duty-cycle percentage (`0..=100`).
    pub duty_cycle_pct: u8,
    /// An optional cron schedule (for [`PolicyMode::Scheduled`]).
    pub schedule: Option<String>,
}

/// How a peer leaves a run (§10.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LeaveMode {
    /// Finish the current round, then leave.
    Graceful,
    /// Leave immediately (abort any in-flight work).
    Immediate,
}

/// A classified worker failure (§10.2) — the swarm analogue of `daemon_infer`'s `ErrorClass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// VRAM/host allocator OOM — worker replaced, micro-batch re-probed.
    OutOfMemory,
    /// A transient network/transport fault — retry in place.
    Transient,
    /// State divergence — the resync path (§9).
    Desync,
    /// An experiment-module trap / sandbox-budget violation — leave the run, worker unharmed (§13).
    Module,
    /// Unrecoverable (crash-loop meltdown, internal bug).
    Fatal,
    /// Cancelled cooperatively.
    Cancelled,
}

/// The worker's capability vocabulary, reported by [`Command::Probe`] (§6.5, §10.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// The tensor-ABI major version the worker implements.
    pub abi_version: u16,
    /// The host-vocabulary ops the worker implements (`name@version`, §5.2).
    pub ops: Vec<String>,
    /// The payload stores the worker can speak (`r2`, `iroh-blobs`, …).
    pub payload_stores: Vec<String>,
}

/// A hardware + capability probe result (§10.2 — extends the daemon-models `HardwareProbe`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    /// The number of usable GPUs.
    pub gpus: u32,
    /// Total VRAM in MiB (across GPUs).
    pub vram_mb: u64,
    /// Installed host RAM in MiB (§5.1 host-RAM planning).
    pub ram_mb: u64,
    /// The backend lanes the worker was built with (`cpu`, `cuda`, `rocm`, `vulkan`).
    pub backend_lanes: Vec<String>,
    /// The capability vocabulary (ABI version, ops, payload stores).
    pub capabilities: WorkerCapabilities,
    /// Measured uplink in kbit/s.
    pub up_kbps: u64,
    /// Measured downlink in kbit/s.
    pub down_kbps: u64,
    /// Free disk for the data/checkpoint cache in MiB.
    pub disk_free_mb: u64,
    /// The measured throughput class (§6.3: `c1`..`c4`).
    pub throughput_class: String,
}

/// A self-assessment result for a run (§6.5, §10.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eligibility {
    /// Whether this peer can join.
    pub eligible: bool,
    /// Human-readable reasons (why-not).
    pub reasons: Vec<String>,
    /// Per-dimension headroom (e.g. `"vram_mb" => 4096`).
    pub headroom: Vec<(String, i64)>,
}

/// A parent → worker command frame (§10.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Command {
    /// Report hardware + capability vocabulary (cached; refreshed on hardware/config change).
    Probe,
    /// Assess a run against this peer's effective resources — read-only, no GPU allocation.
    AssessRun {
        /// The run envelope bytes (opaque here; lane P owns the schema — MERGE-1).
        envelope: Vec<u8>,
    },
    /// Join a run, then stream [`Event`]s.
    JoinRun {
        /// The run to join.
        run_id: String,
        /// The coordinator endpoint (WS/HTTP).
        coordinator: String,
        /// Opaque credentials (daemon-credentials reference / token bytes).
        credentials: Vec<u8>,
        /// The participation policy.
        policy: JoinPolicy,
    },
    /// GPU-governor lever (§10.5). `paused` promises memory, not just time: the worker aborts any
    /// in-flight guest call, drops the wasm instance + GPU allocations, and keeps only CPU masters.
    Throttle {
        /// A VRAM cap in MiB (`None` = unchanged).
        vram_cap_mb: Option<u32>,
        /// A duty-cycle percentage (`None` = unchanged).
        duty_cycle_pct: Option<u8>,
        /// Whether training is paused (preemption-as-churn).
        paused: bool,
    },
    /// Leave a run.
    Leave {
        /// The run to leave.
        run_id: String,
        /// How to leave.
        mode: LeaveMode,
    },
    /// Ask the worker to exit cleanly.
    Shutdown,
    /// Liveness probe (answered with [`Event::Pong`]).
    Ping,
}

/// A worker → parent event frame (§10.2). All are persisted / fanned out by the node (§10.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// The worker started and is ready for commands; reports its capability vocabulary.
    Ready {
        /// The worker's capabilities.
        capabilities: WorkerCapabilities,
    },
    /// A [`Command::Probe`] result.
    Probed(Hardware),
    /// An [`Command::AssessRun`] result.
    Assessed(Eligibility),
    /// The run's phase advanced.
    RunPhase {
        /// The run this phase belongs to.
        run_id: String,
        /// The phase name (`warmup`, `train`, `witness`, `cooldown`, …).
        phase: String,
        /// The current epoch.
        epoch: u64,
        /// The current round.
        round: RoundId,
    },
    /// Progress within a round.
    RoundProgress {
        /// The inner step within the round.
        inner_step: u32,
        /// The current loss.
        loss: f32,
        /// Throughput in tokens/s.
        tokens_per_s: f32,
        /// Bytes uploaded this round so far.
        up_bytes: u64,
        /// Bytes downloaded this round so far.
        down_bytes: u64,
        /// Peers this round involves.
        peers: u32,
    },
    /// The §6.4 protocol as seen from this peer, at round end.
    RoundOutcome {
        /// The round that ended.
        round: RoundId,
        /// The number of payloads committed to the record.
        committed: u32,
        /// The number of payloads this peer ingested.
        ingested: u32,
        /// Whether this peer stalled (missed a committed payload at the barrier).
        stalled: bool,
        /// The post-ingest state digest (§5.6).
        digest: [u8; 16],
    },
    /// A named scalar metric readout.
    Metric {
        /// The metric name.
        name: String,
        /// The metric value.
        value: f64,
    },
    /// A checkpoint was published.
    CheckpointPublished {
        /// The round the checkpoint covers.
        round: RoundId,
        /// The checkpoint's content hash (blake3 hex).
        hash: String,
        /// A locator (store key / blob ticket).
        location: String,
    },
    /// A non-fatal warning (desync-warning, straggling, quota).
    Warning {
        /// The warning class.
        class: String,
        /// A short human-readable detail.
        detail: String,
    },
    /// A classified failure.
    Error {
        /// The failure class (maps to the node's recovery loop).
        class: ErrorClass,
        /// A short human-readable detail.
        detail: String,
    },
    /// Liveness reply to [`Command::Ping`].
    Pong,
}

/// A CBOR codec error (mirrors `daemon_infer::protocol::CodecError`).
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Encoding a frame to CBOR failed.
    #[error("cbor encode: {0}")]
    Encode(String),
    /// Decoding a frame from CBOR failed.
    #[error("cbor decode: {0}")]
    Decode(String),
}

/// Encode a frame body to CBOR bytes (the `CutChannel` adds the length prefix).
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

/// Decode a CBOR frame body.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    ciborium::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_command(cmd: Command) {
        let bytes = encode(&cmd).expect("encode command");
        let back: Command = decode(&bytes).expect("decode command");
        assert_eq!(cmd, back);
    }

    fn round_trip_event(ev: Event) {
        let bytes = encode(&ev).expect("encode event");
        let back: Event = decode(&bytes).expect("decode event");
        assert_eq!(ev, back);
    }

    #[test]
    fn commands_round_trip() {
        round_trip_command(Command::Probe);
        round_trip_command(Command::AssessRun {
            envelope: vec![1, 2, 3, 4],
        });
        round_trip_command(Command::JoinRun {
            run_id: "run-42".into(),
            coordinator: "wss://coord.example/swarm".into(),
            credentials: vec![0xde, 0xad, 0xbe, 0xef],
            policy: JoinPolicy {
                mode: PolicyMode::Idle,
                vram_cap_mb: 12_000,
                duty_cycle_pct: 80,
                schedule: Some("0 2 * * *".into()),
            },
        });
        round_trip_command(Command::Throttle {
            vram_cap_mb: Some(8_000),
            duty_cycle_pct: None,
            paused: true,
        });
        round_trip_command(Command::Leave {
            run_id: "run-42".into(),
            mode: LeaveMode::Graceful,
        });
        round_trip_command(Command::Shutdown);
        round_trip_command(Command::Ping);
    }

    #[test]
    fn events_round_trip() {
        round_trip_event(Event::Ready {
            capabilities: WorkerCapabilities {
                abi_version: 1,
                ops: vec!["matmul@1".into(), "flash_attn@1".into()],
                payload_stores: vec!["r2".into(), "iroh-blobs".into()],
            },
        });
        round_trip_event(Event::Probed(Hardware {
            gpus: 2,
            vram_mb: 24_000,
            ram_mb: 64_000,
            backend_lanes: vec!["cuda".into()],
            capabilities: WorkerCapabilities {
                abi_version: 1,
                ops: vec!["adamw_step@1".into()],
                payload_stores: vec!["r2".into()],
            },
            up_kbps: 50_000,
            down_kbps: 200_000,
            disk_free_mb: 500_000,
            throughput_class: "c3".into(),
        }));
        round_trip_event(Event::Assessed(Eligibility {
            eligible: false,
            reasons: vec!["vram below floor".into()],
            headroom: vec![("vram_mb".into(), -2048), ("ram_mb".into(), 16_000)],
        }));
        round_trip_event(Event::RunPhase {
            run_id: "run-42".into(),
            phase: "train".into(),
            epoch: 3,
            round: 128,
        });
        round_trip_event(Event::RoundProgress {
            inner_step: 12,
            loss: 2.5,
            tokens_per_s: 4200.0,
            up_bytes: 1024,
            down_bytes: 8192,
            peers: 7,
        });
        round_trip_event(Event::RoundOutcome {
            round: 128,
            committed: 6,
            ingested: 6,
            stalled: false,
            digest: [0xAB; 16],
        });
        round_trip_event(Event::Metric {
            name: "grad_norm".into(),
            value: 0.75,
        });
        round_trip_event(Event::CheckpointPublished {
            round: 200,
            hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
            location: "r2://run-42/ckpt-200.safetensors".into(),
        });
        round_trip_event(Event::Warning {
            class: "straggle".into(),
            detail: "late fetch".into(),
        });
        for class in [
            ErrorClass::OutOfMemory,
            ErrorClass::Transient,
            ErrorClass::Desync,
            ErrorClass::Module,
            ErrorClass::Fatal,
            ErrorClass::Cancelled,
        ] {
            round_trip_event(Event::Error {
                class,
                detail: "boom".into(),
            });
        }
        round_trip_event(Event::Pong);
    }
}
