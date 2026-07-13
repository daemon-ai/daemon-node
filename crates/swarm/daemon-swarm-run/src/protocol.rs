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
    /// Dedicated VRAM in MiB (across GPUs). On a unified/integrated GPU this is the small
    /// dedicated carve-out (sysfs `mem_info_vram_total`), NOT the usable budget — that spills into
    /// [`Self::shared_mb`].
    pub vram_mb: u64,
    /// Shared / unified spillover memory in MiB (GTT — sysfs `mem_info_gtt_total`): the host DRAM
    /// an integrated GPU can page tensors into beyond [`Self::vram_mb`]. `0` = none (a classic
    /// discrete GPU). **Additive (Merge 2):** `#[serde(default)]` keeps pre-Merge-2 `Hardware`
    /// payloads (which lack this field) decodable, and a `shared_mb == 0` value serializes
    /// compatibly. This is the worker↔node protocol type; it does NOT cross the SwarmApi wire (the
    /// app-facing DTO is `daemon_api::SwarmHardwareReport`, mapped in the node service), so no CDDL
    /// / wire-version change is implied.
    #[serde(default)]
    pub shared_mb: u64,
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
    /// **Additive (A3, Merge 2).** The autotune micro-batch verdict the worker consumed in-process
    /// for the joined run (the node's `Eligibility.headroom["micro_batch"]`, §10.5). Emitted once
    /// per join by the live-attach path (the P1-deferred telemetry-as-protocol-event follow-on 2;
    /// B3 logged it to stderr). Additive to the frozen §10.2 stream — a new variant only, so every
    /// pre-A3 decoder still round-trips the existing frames.
    MicroBatch {
        /// The chosen micro-batch (sequences per inner step).
        micro_batch: u32,
    },
    /// **Additive (A3, Merge 2).** One rung of the §10.5 OOM-halving ladder: a real `BudgetMemory`
    /// trap forced the worker to churn the instance and retry at half the micro-batch. Emitted by
    /// the live-attach path when the ladder fires (B3 logged it to stderr). Additive to the frozen
    /// §10.2 stream.
    OomLadder {
        /// The round the ladder fired on.
        round: RoundId,
        /// The micro-batch before halving.
        from_micro_batch: u32,
        /// The micro-batch after halving.
        to_micro_batch: u32,
        /// Cumulative halvings on this round so far.
        halvings: u32,
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

// ---------------------------------------------------------------------------
// JoinRun.credentials contract (A3, frozen at Merge 2)
// ---------------------------------------------------------------------------
//
// `Command::JoinRun.credentials` stays an OPAQUE `Vec<u8>` on the frozen worker wire (§10.2). A3
// defines the canonical-CBOR **schema** carried in it: the node's `SwarmService` / run-authoring
// path AUTHORS a [`JoinCredentials`], `to_bytes()` it into `credentials`, and the worker's live
// attach `from_bytes()` it to construct the live plane + engine. It is a NEW additive type — no
// `Command`/`Event` shape change — so a worker built without the live-attach feature ignores the
// bytes exactly as before.

/// The WS coordinator auth for the live attach (the canonical-CBOR mirror of
/// `daemon_swarm_net::ws_client::WsAuth`, defined here so the dependency-light protocol crate owns
/// the credentials schema; the worker converts it under the `ws` feature — never hardcoded).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WsAuthSpec {
    /// No auth (bare `ws://` dev target).
    #[default]
    None,
    /// `Authorization: Bearer <token>` (the gateway `swarm:join` path).
    Bearer(String),
    /// The internal identity headers `x-daemon-org-id` / `x-daemon-actor` (direct-to-`apps/swarm`).
    Internal {
        /// The org id header value.
        org_id: String,
        /// The actor header value.
        actor: String,
    },
}

/// One iroh roster peer (the canonical-CBOR mirror of `daemon_swarm_net::IrohPeer` for the
/// credentials body: an `endpoint_id` + reachability). Direct addrs are `"ip:port"` strings so the
/// protocol crate takes no `std::net` serialization dependency; the worker parses them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrohRosterPeer {
    /// The peer's iroh `EndpointId` (32 raw bytes).
    pub endpoint_id: [u8; 32],
    /// Direct socket addresses (`"ip:port"`); may be empty for relay-only reachability.
    #[serde(default)]
    pub direct_addrs: Vec<String>,
    /// Home relay URL (NAT-proof reachability); `None` for direct-only (LAN/loopback).
    #[serde(default)]
    pub relay_url: Option<String>,
}

/// The optional iroh half of the credentials. Present ⇒ the worker builds a
/// `DualPlane(WsControlPlane, IrohGossip)`; absent ⇒ WS-only (the T0 baseline).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrohCredentials {
    /// The iroh secret key (32 bytes) — separate from the node ed25519 identity (§7.2).
    pub secret_key: [u8; 32],
    /// Envelope-pinned relay URLs (empty ⇒ direct-only / loopback).
    #[serde(default)]
    pub relay_urls: Vec<String>,
    /// The bootstrap roster (may be empty; roster updates arrive dynamically as coordinator frames
    /// and are wired to `IrohGossip::update_roster`).
    #[serde(default)]
    pub roster: Vec<IrohRosterPeer>,
}

/// The engine + corpus knobs the worker's `RoundEngine` needs (from the run's declared config /
/// frozen envelope). Deterministic across peers so the digest transcript agrees.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineParams {
    /// Inner steps per round (§5.1 cadence).
    pub steps_per_round: u32,
    /// Micro-batch (sequences) within an inner step (the assess verdict overrides this at runtime).
    pub micro_batch: u32,
    /// Fetch-recovery budget before a stalled peer leaves for the epoch (§6.4).
    pub stall_rounds_max: u32,
    /// Round-boundary checkpoint cadence (§9); `0` disables.
    pub checkpoint_every_rounds: u32,
    /// §7.3 receive-side per-peer payload cap in bytes (`0` = uncapped) — the worker mirrors the DO
    /// shell's pre-filter (Merge-1 Decision 2).
    #[serde(default)]
    pub update_max_bytes: u64,
    /// Synthetic-corpus seed (deterministic training data — identical across peers → agreeing
    /// digests). Mirrors `daemon_swarm_run::data::Corpus::synthetic(seed, shards, tokens_per_shard,
    /// seq_len)`.
    pub corpus_seed: u64,
    /// Synthetic-corpus shard count.
    pub corpus_shards: u32,
    /// Synthetic-corpus tokens per shard.
    pub corpus_tokens_per_shard: u64,
    /// Synthetic-corpus sequence length (tokens).
    pub corpus_seq_len: u32,
    /// Clamp corpus token ids into the experiment's vocabulary (`token % clamp`; `0` = no clamp) —
    /// the deterministic per-token stand-in for tokenizing the corpus at the model's vocab (the B3
    /// live-e2e shim recipe, applied identically by every peer so digests agree).
    #[serde(default)]
    pub corpus_vocab_clamp: u32,
}

/// The canonical-CBOR body of [`Command::JoinRun`]'s `credentials` (A3, frozen at Merge 2). Authored
/// node-side, parsed by the worker's live attach.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinCredentials {
    /// The peer's node ed25519 signing-key seed (32 bytes) — the `RoundEngine`'s `Join` signer
    /// identity (§7.2). Also the `PeerId` this peer contributes under.
    pub node_secret: [u8; 32],
    /// WS coordinator auth.
    #[serde(default)]
    pub ws_auth: WsAuthSpec,
    /// The epoch roster (node pubkeys, 32-byte) the engine folds each round.
    pub roster: Vec<[u8; 32]>,
    /// blake3 of the frozen envelope (§6.1) — the iroh topic-derivation input + admission binding.
    pub envelope_hash: [u8; 32],
    /// Optional iroh transport (dual-plane). Absent ⇒ WS-only (T0 baseline).
    #[serde(default)]
    pub iroh: Option<IrohCredentials>,
    /// Optional presign base for the `R2Store` payload plane (e.g.
    /// `http://127.0.0.1:8795/api/v1/swarm`). Absent ⇒ `FsPayloadStore` fallback (tests / LAN).
    #[serde(default)]
    pub presign_base: Option<String>,
    /// The engine + corpus knobs.
    pub engine: EngineParams,
}

impl JoinCredentials {
    /// Encode to the canonical-CBOR bytes carried in `JoinRun.credentials`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CodecError> {
        encode(self)
    }

    /// Decode from `JoinRun.credentials` bytes. An empty / non-`JoinCredentials` buffer decodes to
    /// an error, which the worker treats as "no live attach" (the self-driven fallback).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        decode(bytes)
    }
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
            shared_mb: 120_000,
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
        // A3 additive telemetry variants.
        round_trip_event(Event::MicroBatch { micro_batch: 4 });
        round_trip_event(Event::OomLadder {
            round: 7,
            from_micro_batch: 4,
            to_micro_batch: 2,
            halvings: 1,
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

    /// `join_credentials_round_trip`: the A3 `JoinCredentials` schema carried in
    /// `JoinRun.credentials` round-trips through the canonical-CBOR codec, incl. the optional iroh /
    /// presign halves and the WS-auth variants.
    #[test]
    fn join_credentials_round_trip() {
        let creds = JoinCredentials {
            node_secret: [0x11; 32],
            ws_auth: WsAuthSpec::Internal {
                org_id: "org_live".into(),
                actor: "key:live".into(),
            },
            roster: vec![[0x22; 32], [0x33; 32]],
            envelope_hash: [0xEE; 32],
            iroh: Some(IrohCredentials {
                secret_key: [0x44; 32],
                relay_urls: vec!["http://127.0.0.1:3340".into()],
                roster: vec![IrohRosterPeer {
                    endpoint_id: [0x55; 32],
                    direct_addrs: vec!["127.0.0.1:4550".into()],
                    relay_url: Some("http://127.0.0.1:3340".into()),
                }],
            }),
            presign_base: Some("http://127.0.0.1:8795/api/v1/swarm".into()),
            engine: EngineParams {
                steps_per_round: 2,
                micro_batch: 2,
                stall_rounds_max: 3,
                checkpoint_every_rounds: 0,
                update_max_bytes: 1 << 20,
                corpus_seed: 7,
                corpus_shards: 4,
                corpus_tokens_per_shard: 256,
                corpus_seq_len: 8,
                corpus_vocab_clamp: 64,
            },
        };
        let bytes = creds.to_bytes().expect("encode credentials");
        let back = JoinCredentials::from_bytes(&bytes).expect("decode credentials");
        assert_eq!(creds, back);

        // The WS-only baseline (no iroh, no presign) also round-trips, and a non-credentials buffer
        // is a decode error (the worker's "no live attach → self-driven fallback" signal).
        let ws_only = JoinCredentials {
            iroh: None,
            presign_base: None,
            ws_auth: WsAuthSpec::None,
            ..creds
        };
        let back2 = JoinCredentials::from_bytes(&ws_only.to_bytes().unwrap()).unwrap();
        assert_eq!(ws_only, back2);
        assert!(JoinCredentials::from_bytes(&[]).is_err());
    }

    /// `hardware_shared_mb_is_additive_back_compatible`: the Merge-2 `shared_mb` field is additive.
    /// A pre-Merge-2 `Hardware` payload (a CBOR map WITHOUT `shared_mb`) still decodes, with
    /// `shared_mb` defaulting to 0; and a `shared_mb == 0` value is carried through a round-trip.
    #[test]
    fn hardware_shared_mb_is_additive_back_compatible() {
        // A pre-Merge-2 `Hardware` had no `shared_mb`. Model it with a mirror struct and decode the
        // legacy bytes into the current type: `#[serde(default)]` fills `shared_mb = 0`.
        #[derive(serde::Serialize)]
        struct LegacyHardware {
            gpus: u32,
            vram_mb: u64,
            ram_mb: u64,
            backend_lanes: Vec<String>,
            capabilities: WorkerCapabilities,
            up_kbps: u64,
            down_kbps: u64,
            disk_free_mb: u64,
            throughput_class: String,
        }
        let legacy = LegacyHardware {
            gpus: 1,
            vram_mb: 4096,
            ram_mb: 124_419,
            backend_lanes: vec!["vulkan".into(), "cpu".into()],
            capabilities: WorkerCapabilities::default(),
            up_kbps: 0,
            down_kbps: 0,
            disk_free_mb: 0,
            throughput_class: "c1".into(),
        };
        let bytes = encode(&legacy).expect("encode legacy");
        let decoded: Hardware = decode(&bytes).expect("legacy Hardware still decodes");
        assert_eq!(decoded.shared_mb, 0, "missing field defaults to 0");
        assert_eq!(decoded.vram_mb, 4096);

        // Full round-trip preserves a real GTT number.
        let hw = Hardware {
            gpus: 1,
            vram_mb: 4096,
            shared_mb: 120_000,
            ram_mb: 124_419,
            ..Hardware::default()
        };
        let back: Hardware = decode(&encode(&hw).expect("encode")).expect("decode");
        assert_eq!(back.shared_mb, 120_000);
    }
}
