// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **C3 heterogeneous-fleet live-attach harness — the P2 WAN-gate heterogeneity rehearsal.**
//!
//! The Merge-2 `ws_live_workers.rs` proved the full node↔cloud↔worker loop + churn drill with N
//! **local** `daemon-train-worker` subprocesses (all one platform). This harness proves the *new*
//! P2-gate property: peers on **different platforms / GPU vendors** completing the same run reach
//! **byte-identical det-lane digests** — the consensus bar (spec §5.6: the det lane is CPU fp32 by
//! contract, so a cross-compiled worker on real Windows/macOS/CUDA must agree bit-for-bit with the
//! Linux peer). Divergence on the *native/GPU* lane is allowed and recorded; det-lane equality is the
//! gate.
//!
//! Peers are spawned via `TrainSupervisor`, but each peer's spawn command is **configurable**, so a
//! peer can be a local binary OR a remote process reached over `ssh` (the worker speaks length-framed
//! CBOR over stdin/stdout — `ssh -T <box> <remote-exe>` pipes it binary-clean; validated on the
//! Windows 5090 box). This is the same harness the gate operator drives against the real fleet.
//!
//! Env (skips cleanly when `SWARM_FLEET_WS_URL` is unset — offline gate):
//! ```text
//! SWARM_FLEET_WS_URL       coordinator/registry base (e.g. https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm)
//! SWARM_FLEET_PRESIGN_BASE presign base (default: SWARM_FLEET_WS_URL)
//! SWARM_FLEET_RELAY_URL    optional iroh relay URL — set ⇒ dual-plane (WS + iroh); unset ⇒ WS-only
//! SWARM_FLEET_ORG/ACTOR    internal identity headers (default org_live / key:live)
//! SWARM_FLEET_ROUNDS       rounds to run (default 6)
//! SWARM_FLEET_PEERS        peer spec list (see below). Unset ⇒ two LOCAL peers (a self-check).
//! ```
//!
//! `SWARM_FLEET_PEERS` is a `;;`-separated list; each entry is `label|program|arg0|arg1|…`. The
//! special program `LOCAL` means "the locally-built `daemon-train-worker` with `DAEMON_TRAIN_MODULE`
//! set to the local guest". Any other program is spawned verbatim (args follow) — for a remote peer,
//! bake the module env into the command. Example (Windows 5090 over ssh + one local Linux peer):
//! ```text
//! SWARM_FLEET_PEERS='linux-vulkan|LOCAL;;win-5090|ssh|-T|usergpu356@37.230.134.194|set DAEMON_TRAIN_MODULE=C:\Users\Administrator\tiny_llama.wasm && daemon-train-worker.exe'
//! ```
//!
//! Drive it:
//! ```text
//! SWARM_FLEET_WS_URL=https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm \
//! SWARM_FLEET_PEERS='…' cargo test -p daemon-swarm-e2e --features iroh \
//!   --test fleet_live_hetero -- --nocapture --test-threads 1
//! ```

// Test harness: builds the worker binary + guests via cargo (the sanctioned dev-tool exception,
// mirroring `ws_live_workers.rs`), reads operator-local files, spawns ssh, and prints progress.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_swarm_net::{ControlPlane, ReconnectConfig, WsAuth, WsConfig, WsControlPlane};
use daemon_swarm_observe::desync::digest_tally_from_log;
use daemon_swarm_observe::{MessageLog, RunHealth};
use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_swarm_proto::{
    from_canonical_slice, peer_id, to_canonical_vec, SignedMessage, SigningKey,
};
use daemon_swarm_run::protocol::{
    CorpusRef, EngineParams, Event, IrohCredentials, IrohRosterPeer, JoinCredentials, JoinPolicy,
    LeaveMode, PolicyMode, WsAuthSpec,
};
use daemon_train_client::{TrainClientConfig, TrainSupervisor};
use daemon_train_sdk::models::TinyLlamaCfg;

// ---- P3 Merge-2: the 160M content-addressed ceremony mode ----------------------------------------
//
// The P2 ceremony (below) ran tiny-llama with a synthetic corpus and pre-staged the module via
// `DAEMON_TRAIN_MODULE`. The program gate converts that into the real 160M fleet measurement: the
// experiment module + the tokenized TinyStories corpus are fetched BY CONTENT HASH from the payload
// store (lane S). When the `SWARM_GATE_MODULE_BLAKE3` + `SWARM_GATE_MANIFEST_BLAKE3` env are set, the
// shared authoring below emits the content-addressed 160M envelope + `EngineParams.corpus`, and LOCAL
// peers carry the `DAEMON_SWARM_*` fetch context + `DAEMON_TRAIN_BACKEND` instead of a module path
// (remote peers carry that env in their ssh command, per the artifact-distribution runbook §3-§4).
// Absent that env, everything is byte-identical to the P2 tiny-llama path.

/// The content-addressed 160M ceremony assets (lane S published), from env. `None` ⇒ the P2
/// tiny-llama synthetic-corpus path (unchanged).
struct GateAssets {
    module_blake3: [u8; 32],
    module_size: u64,
    manifest_blake3: [u8; 32],
    manifest_size: u64,
    asset_scope: String,
    backend: String,
    reduced: bool,
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn gate_assets() -> Option<GateAssets> {
    let module_blake3 = hex32(&std::env::var("SWARM_GATE_MODULE_BLAKE3").ok()?)?;
    let manifest_blake3 = hex32(&std::env::var("SWARM_GATE_MANIFEST_BLAKE3").ok()?)?;
    Some(GateAssets {
        module_blake3,
        module_size: env_u64("SWARM_GATE_MODULE_SIZE", 0),
        manifest_blake3,
        manifest_size: env_u64("SWARM_GATE_MANIFEST_SIZE", 0),
        asset_scope: std::env::var("SWARM_GATE_ASSET_SCOPE")
            .unwrap_or_else(|_| "assets-p3s".into()),
        backend: std::env::var("SWARM_GATE_BACKEND").unwrap_or_else(|_| "wgpu".into()),
        reduced: std::env::var_os("SWARM_GATE_REDUCED").is_some(),
    })
}

fn gate_cfg(g: &GateAssets) -> TinyLlamaCfg {
    if g.reduced {
        TinyLlamaCfg {
            n_layers: 1,
            d_model: 256,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 64,
            vocab: 50257,
            seq_len: 1024,
            ..TinyLlamaCfg::llama_160m()
        }
    } else {
        TinyLlamaCfg::llama_160m()
    }
}

const NUM_ROUNDS_DEFAULT: u64 = 6;
const GUEST_VOCAB: u32 = 64;
// Declared RunConfig phase timings. Env-overridable (`SWARM_FLEET_{WARMUP,ROUND_TIMEOUT,COOLDOWN}_S`)
// because the barrier needs a round window that comfortably exceeds the SLOWEST peer's per-round wall
// (`min_peers == N` ⇒ every peer must commit each round or the floor breaches and the run parks —
// and this lean harness, unlike ws_live_workers, has no rejoin). Defaults suit LAN/local; a live WAN
// run with a Windows-over-ssh + macOS-nix-develop-over-ssh peer wants a larger round timeout.
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn declared_warmup_s() -> u64 {
    env_u64("SWARM_FLEET_WARMUP_S", 12)
}
fn declared_round_timeout_s() -> u64 {
    env_u64("SWARM_FLEET_ROUND_TIMEOUT_S", 45)
}
fn declared_cooldown_s() -> u64 {
    env_u64("SWARM_FLEET_COOLDOWN_S", 2)
}
// Must divide across `peers × steps_per_round × micro_batch`; set from the peer count at run time.
// Data-partition shape. Env-overridable (`SWARM_FLEET_STEPS_PER_ROUND` / `SWARM_FLEET_MICRO_BATCH`,
// default 2/2 = the P2 tiny-llama shape) so the 160M ceremony can dial them to 1/1 — a CPU-det peer
// (the sealed Windows cross-build carries no GPU backend) then does one 160M step/round, tractable
// within the barrier round timeout.
fn steps_per_round() -> u32 {
    env_u64("SWARM_FLEET_STEPS_PER_ROUND", 2) as u32
}
fn micro_batch() -> u32 {
    env_u64("SWARM_FLEET_MICRO_BATCH", 2) as u32
}
// §9 checkpoint cadence + resync-replay window used by the churn ceremony (lane R): a checkpoint
// every 2 rounds guarantees one is published + registered before the default kill (after round 2),
// so the rejoiner resumes from it. Retention comfortably exceeds any short-run replay gap.
const CHECKPOINT_EVERY_ROUNDS: u32 = 2;
/// Env-overridable checkpoint cadence (`SWARM_FLEET_CHECKPOINT_EVERY`, default 2; `0` = never).
/// Diagnostic knob (P3 Merge-2): the 160M CUDA-live pod panic fires exactly at the first
/// checkpoint round — `checkpoint_save`'s device readback staging on a cubecl pool already at
/// ~24.1/24.5 GiB. Setting 0 isolates the trigger. The churn ceremony REQUIRES a checkpoint
/// (resync drill), so this stays 2 there.
fn checkpoint_every_rounds() -> u32 {
    env_u64(
        "SWARM_FLEET_CHECKPOINT_EVERY",
        u64::from(CHECKPOINT_EVERY_ROUNDS),
    ) as u32
}
const PAYLOAD_RETENTION_ROUNDS: u64 = 16;

// ---- env config ----------------------------------------------------------------------------------

struct FleetEnv {
    ws_base: String,
    presign_base: String,
    relay_url: Option<String>,
    org: String,
    actor: String,
    rounds: u64,
}

fn fleet_env() -> Option<FleetEnv> {
    let ws_base = std::env::var("SWARM_FLEET_WS_URL").ok()?;
    Some(FleetEnv {
        presign_base: std::env::var("SWARM_FLEET_PRESIGN_BASE").unwrap_or_else(|_| ws_base.clone()),
        relay_url: std::env::var("SWARM_FLEET_RELAY_URL").ok(),
        org: std::env::var("SWARM_FLEET_ORG").unwrap_or_else(|_| "org_live".into()),
        actor: std::env::var("SWARM_FLEET_ACTOR").unwrap_or_else(|_| "key:live".into()),
        rounds: std::env::var("SWARM_FLEET_ROUNDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(NUM_ROUNDS_DEFAULT),
        ws_base,
    })
}

/// A peer's label + spawn command. `LOCAL` is expanded to the built worker + module env.
struct PeerSpec {
    label: String,
    program: String,
    args: Vec<String>,
    is_local: bool,
}

fn parse_peers(local_bin: &Path, module_path: &Path) -> Vec<PeerSpec> {
    let raw =
        std::env::var("SWARM_FLEET_PEERS").unwrap_or_else(|_| "peer-a|LOCAL;;peer-b|LOCAL".into());
    raw.split(";;")
        .filter(|e| !e.trim().is_empty())
        .map(|entry| {
            let mut parts = entry.split('|');
            let label = parts.next().unwrap_or("peer").to_string();
            let program = parts.next().unwrap_or("LOCAL").to_string();
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();
            let is_local = program == "LOCAL";
            if is_local {
                PeerSpec {
                    label,
                    program: local_bin.to_string_lossy().into_owned(),
                    args: Vec::new(),
                    is_local,
                }
            } else {
                let _ = module_path;
                PeerSpec {
                    label,
                    program,
                    args,
                    is_local,
                }
            }
        })
        .collect()
}

// ---- guest + local worker binary build (mirrors ws_live_workers.rs) -------------------------------

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn guests_root() -> PathBuf {
    workspace_root().join("guests")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

fn ensure_built() -> PathBuf {
    // Operator-supplied prebuilt worker (P3 Merge-2): the 160M ceremony spawns a RELEASE local
    // worker (debug cubecl at 160M is slow + memory-hungry); the harness itself stays a debug test
    // binary. When set, skip the cargo build entirely and trust the given path.
    if let Ok(bin) = std::env::var("SWARM_FLEET_LOCAL_BIN") {
        let bin = PathBuf::from(bin);
        assert!(bin.exists(), "SWARM_FLEET_LOCAL_BIN at {}", bin.display());
        return bin;
    }
    BUILD.call_once(|| {
        // The LOCAL peer's feature set. Default `swarm-net` (the P2 tiny-llama CPU-det path); the P3
        // 160M ceremony sets `SWARM_FLEET_WORKER_FEATURES=swarm-net,wgpu` so the local Strix/RADV peer
        // runs the wgpu native lane (160M on the CPU det lane is intractably slow — S deviation 1).
        let features = std::env::var("SWARM_FLEET_WORKER_FEATURES")
            .unwrap_or_else(|_| "swarm-net".to_string());
        let status = Command::new("cargo")
            .current_dir(workspace_root())
            .args([
                "build",
                "-p",
                "daemon-train",
                "--features",
                &features,
                "--bin",
                "daemon-train-worker",
            ])
            .status()
            .expect("run cargo for the live worker binary");
        assert!(status.success(), "building daemon-train-worker failed");
    });
    let target = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("target"));
    let bin = target.join("debug/daemon-train-worker");
    assert!(bin.exists(), "worker binary at {}", bin.display());
    bin
}

fn tiny_llama_wasm_path() -> PathBuf {
    let path = guest_dir().join("tiny_llama.wasm");
    assert!(path.exists(), "guest module at {}", path.display());
    path
}

// ---- run authoring (the §6.1 chain) --------------------------------------------------------------

fn tiny_cfg() -> ciborium::value::Value {
    ciborium::value::Value::serialized(&TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        vocab: GUEST_VOCAB,
        profile: "sparse_loco".to_string(),
        ..TinyLlamaCfg::default()
    })
    .expect("tiny-llama config serializes")
}

fn author_envelope(
    run_id: &str,
    module_path: &Path,
    module_bytes: &[u8],
    peers: u32,
    rounds: u64,
    global_batch: u32,
) -> Envelope {
    // P3 gate mode: content-addressed 160M module + corpus manifest, fetched by hash (no local path).
    let (artifacts, module_name, manifest_name, config, update_mb_max) =
        if let Some(g) = gate_assets() {
            let module_hex: String = g.module_blake3.iter().map(|b| format!("{b:02x}")).collect();
            let manifest_hex: String = g
                .manifest_blake3
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let mut artifacts = std::collections::BTreeMap::new();
            artifacts.insert(
                "experiment.wasm".to_string(),
                Artifact {
                    url: format!("r2://modules/{module_hex}.wasm"),
                    blake3: daemon_swarm_proto::Hash::new(g.module_blake3),
                },
            );
            artifacts.insert(
                "data.manifest".to_string(),
                Artifact {
                    url: format!("r2://corpus/{manifest_hex}.json"),
                    blake3: daemon_swarm_proto::Hash::new(g.manifest_blake3),
                },
            );
            (
                artifacts,
                "experiment.wasm".to_string(),
                "data.manifest".to_string(),
                cfg_value(&gate_cfg(&g)),
                64u32,
            )
        } else {
            let mut artifacts = std::collections::BTreeMap::new();
            artifacts.insert(
                "tiny_llama.wasm".to_string(),
                Artifact {
                    url: format!("file://{}", module_path.display()),
                    blake3: daemon_swarm_proto::blake3_hash(module_bytes),
                },
            );
            (
                artifacts,
                "tiny_llama.wasm".to_string(),
                "tiny_llama.wasm".to_string(),
                tiny_cfg(),
                1u32,
            )
        };
    Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: run_id.to_string(),
            min_peers: peers,
            max_peers: peers + 1,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: module_name,
            abi: "tabi@1".to_string(),
            config,
        },
        artifacts,
        data: DataSection {
            manifest: manifest_name,
            steps_per_round: steps_per_round(),
            global_batch: GlobalBatch {
                start: global_batch,
                end: global_batch,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(rounds),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 0,
            uplink_mbps_min: 0,
            downlink_mbps_min: 0,
            disk_gb_min: 0,
            throughput_floor: "c1".to_string(),
            update_mb_max,
            capabilities: Vec::new(),
            payload_store: "r2".to_string(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: declared_warmup_s() as u32,
            round_train_max: declared_round_timeout_s() as u32,
            round_witness: 30,
            cooldown: declared_cooldown_s() as u32,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 3,
            payload_retention_rounds: 16,
        },
    }
}

fn cfg_value(cfg: &TinyLlamaCfg) -> ciborium::value::Value {
    ciborium::value::Value::serialized(cfg).expect("preset config serializes")
}

// ---- registry HTTP -------------------------------------------------------------------------------

async fn post_json(
    egress: &EgressClient,
    env: &FleetEnv,
    url: &str,
    body: &serde_json::Value,
) -> (u16, String) {
    let req = EgressRequest::post_json(url, body)
        .expect("encode request")
        .header("x-daemon-org-id", &env.org)
        .header("x-daemon-actor", &env.actor);
    let resp = egress
        .execute(req, Redirects::None)
        .await
        .expect("registry POST");
    let status = resp.status().as_u16();
    let text = String::from_utf8_lossy(&resp.bytes().await.expect("read body")).into_owned();
    (status, text)
}

async fn get_json(egress: &EgressClient, env: &FleetEnv, url: &str) -> (u16, serde_json::Value) {
    let req = EgressRequest::get(url)
        .header("x-daemon-org-id", &env.org)
        .header("x-daemon-actor", &env.actor);
    let resp = egress
        .execute(req, Redirects::None)
        .await
        .expect("registry GET");
    let status = resp.status().as_u16();
    let body = resp.bytes().await.expect("read body");
    let value = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    (status, value)
}

fn create_run_request(
    envelope: &Envelope,
    frozen: &daemon_swarm_proto::FrozenEnvelope,
    module_bytes: &[u8],
    rounds: u64,
    global_batch: u32,
) -> serde_json::Value {
    use base64::Engine as _;
    let artifacts = if let Some(g) = gate_assets() {
        let module_hex: String = g.module_blake3.iter().map(|b| format!("{b:02x}")).collect();
        let manifest_hex: String = g
            .manifest_blake3
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        serde_json::json!([
            { "path": format!("modules/{module_hex}.wasm"), "blake3": module_hex, "size": g.module_size },
            { "path": format!("corpus/{manifest_hex}.json"), "blake3": manifest_hex, "size": g.manifest_size },
        ])
    } else {
        serde_json::json!([{
            "path": "tiny_llama.wasm",
            "blake3": daemon_swarm_proto::blake3_hash(module_bytes).to_hex(),
            "size": module_bytes.len(),
        }])
    };
    serde_json::json!({
        "run_id": envelope.run.run_id,
        "schema": ENVELOPE_SCHEMA_MAJOR,
        "proto_version": daemon_swarm_proto::SWARM_PROTO_VERSION,
        "envelope_b64": base64::engine::general_purpose::STANDARD.encode(frozen.bytes()),
        "author_pubkey": frozen.signer().to_hex(),
        "signature": frozen.signature().to_hex(),
        "artifacts": artifacts,
        "update_max_bytes": u64::from(envelope.requirements.update_mb_max) * 1024 * 1024,
        "min_peers": envelope.run.min_peers,
        "max_peers": envelope.run.max_peers,
        "rounds": rounds,
        "warmup_timeout_s": declared_warmup_s(),
        "round_timeout_s": declared_round_timeout_s(),
        "cooldown_s": declared_cooldown_s(),
        "global_batch": global_batch,
        "witness_target": 0,
    })
}

// ---- credentials ---------------------------------------------------------------------------------

fn node_secret(i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0x51 + i as u8;
    s[1] = 0xC3;
    s
}

fn iroh_secret(i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0xB0 + i as u8;
    s[1] = 0xC3;
    s
}

fn credentials_for(
    i: usize,
    n: usize,
    env: &FleetEnv,
    envelope_hash: [u8; 32],
    global_batch: u32,
) -> JoinCredentials {
    let roster: Vec<[u8; 32]> = (0..n)
        .map(|j| peer_id(&SigningKey::from_bytes(&node_secret(j))).0)
        .collect();
    let iroh = env.relay_url.as_ref().map(|relay| IrohCredentials {
        secret_key: iroh_secret(i),
        relay_urls: vec![relay.clone()],
        roster: (0..n)
            .map(|j| IrohRosterPeer {
                endpoint_id: peer_id(&SigningKey::from_bytes(&iroh_secret(j))).0,
                direct_addrs: Vec::new(),
                relay_url: Some(relay.clone()),
            })
            .collect(),
    });
    // P3 gate mode: the run consumes the real TinyStories corpus by content hash over an active
    // window sized to the run (rounds × global_batch sequences). Absent gate assets ⇒ the synthetic
    // corpus (unchanged). `corpus_vocab_clamp = 0` in gate mode: real GPT-2 BPE tokens are in-vocab.
    let (corpus, vocab_clamp, update_max_bytes) = match gate_assets() {
        Some(g) => (
            Some(CorpusRef {
                manifest_blake3: g.manifest_blake3,
                manifest_size: g.manifest_size,
                window_start: 0,
                window_sequences: env.rounds * u64::from(global_batch),
            }),
            0,
            64 << 20,
        ),
        None => (None, GUEST_VOCAB, 1 << 20),
    };
    JoinCredentials {
        node_secret: node_secret(i),
        ws_auth: WsAuthSpec::Internal {
            org_id: env.org.clone(),
            actor: env.actor.clone(),
        },
        roster,
        envelope_hash,
        iroh,
        presign_base: Some(env.presign_base.clone()),
        engine: EngineParams {
            steps_per_round: steps_per_round(),
            micro_batch: micro_batch(),
            stall_rounds_max: 3,
            // Checkpoint every 2 rounds (§9) so a registered checkpoint exists BEFORE the churn kill
            // (drop_after_round default 2 ⇒ a post-round-1 checkpoint is published first) — the
            // rejoiner then resumes from it + replays retained rounds (lane R live resync).
            checkpoint_every_rounds: checkpoint_every_rounds(),
            update_max_bytes,
            corpus_seed: 7,
            corpus_shards: 4,
            corpus_tokens_per_shard: 256,
            corpus_seq_len: 8,
            corpus_vocab_clamp: vocab_clamp,
            // §9 resync-replay window — mirrors the envelope's `payload_retention_rounds`.
            payload_retention_rounds: PAYLOAD_RETENTION_ROUNDS,
            corpus,
        },
    }
}

/// Push the P3 gate-mode fetch context (`DAEMON_SWARM_*`) + `DAEMON_TRAIN_BACKEND` onto a LOCAL peer's
/// worker env instead of `DAEMON_TRAIN_MODULE` — the module + corpus travel by content hash from the
/// store (lane S). Remote (ssh) peers carry this env in their spawn command (runbook §3-§4), so this
/// only touches LOCAL peers. Returns true iff gate mode is active (caller then skips DAEMON_TRAIN_MODULE).
fn push_gate_fetch_env(cfg: &mut TrainClientConfig, env: &FleetEnv, cache_tag: &str) -> bool {
    let Some(g) = gate_assets() else {
        return false;
    };
    let cache_dir = std::env::temp_dir().join(cache_tag);
    cfg.env
        .push(("DAEMON_SWARM_PRESIGN_BASE".into(), env.presign_base.clone()));
    cfg.env
        .push(("DAEMON_SWARM_RUN_ID".into(), g.asset_scope.clone()));
    cfg.env.push(("DAEMON_SWARM_ORG".into(), env.org.clone()));
    cfg.env
        .push(("DAEMON_SWARM_ACTOR".into(), env.actor.clone()));
    cfg.env.push((
        "DAEMON_SWARM_CACHE_DIR".into(),
        cache_dir.to_string_lossy().into_owned(),
    ));
    cfg.env
        .push(("DAEMON_TRAIN_BACKEND".into(), g.backend.clone()));
    true
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

// ---- the rehearsal -------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fleet_heterogeneous_det_lane_agrees() {
    let Some(env) = fleet_env() else {
        eprintln!(
            "SKIP fleet_live_hetero: set SWARM_FLEET_WS_URL (+ SWARM_FLEET_PEERS) — see the module docs"
        );
        return;
    };
    let local_bin = ensure_built();
    let module_path = tiny_llama_wasm_path();
    let module_bytes = std::fs::read(&module_path).expect("read tiny_llama.wasm");

    let peers = parse_peers(&local_bin, &module_path);
    let n = peers.len();
    assert!(
        n >= 2,
        "need >= 2 peers for a cross-peer agreement assertion"
    );
    let global_batch = n as u32 * steps_per_round() * micro_batch();
    let rounds = env.rounds;
    println!(
        "fleet rehearsal: {n} peers {:?}, {rounds} rounds, WS{}",
        peers.iter().map(|p| &p.label).collect::<Vec<_>>(),
        if env.relay_url.is_some() {
            "+iroh"
        } else {
            "-only"
        }
    );

    let run_id = format!(
        "run-c3-fleet-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs()
    );

    let envelope = author_envelope(
        &run_id,
        &module_path,
        &module_bytes,
        n as u32,
        rounds,
        global_batch,
    );
    envelope.validate().expect("envelope validates");
    let author = SigningKey::from_bytes(&[0xC3u8; 32]);
    let frozen = envelope.freeze(&author).expect("freeze envelope");
    frozen.verify().expect("verify frozen envelope");
    let envelope_hash = frozen.hash().0;
    let wire = to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope");

    let egress = EgressClient::new(EgressConfig::default()).expect("egress client");
    let request = create_run_request(&envelope, &frozen, &module_bytes, rounds, global_batch);
    let (status, text) = post_json(&egress, &env, &format!("{}/runs", env.ws_base), &request).await;
    assert_eq!(status, 201, "POST /runs: {text}");
    println!("created {run_id}");

    // Spawn each peer, assess, then join_streaming; collect its RoundOutcome digests.
    let mut supervisors = Vec::new();
    let mut streams = Vec::new();
    for (i, peer) in peers.iter().enumerate() {
        let mut cfg = TrainClientConfig::new(&peer.program);
        cfg.args = peer.args.clone();
        if peer.is_local && !push_gate_fetch_env(&mut cfg, &env, "daemon-swarm-cache-fleet") {
            cfg.env.push((
                "DAEMON_TRAIN_MODULE".to_string(),
                module_path.to_string_lossy().into_owned(),
            ));
        }
        cfg.spawn_timeout = Duration::from_secs(90);
        cfg.op_timeout = Duration::from_secs(600);
        let sup = Arc::new(TrainSupervisor::new(cfg));
        let elig = sup
            .assess(wire.clone())
            .await
            .unwrap_or_else(|e| panic!("peer {} ({}) assess: {e}", i, peer.label));
        assert!(
            elig.eligible,
            "peer {} ({}) eligible: {:?}",
            i, peer.label, elig.reasons
        );
        let creds = credentials_for(i, n, &env, envelope_hash, global_batch)
            .to_bytes()
            .expect("encode credentials");
        let rx = sup
            .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
            .await
            .unwrap_or_else(|e| panic!("peer {} ({}) join_streaming: {e}", i, peer.label));
        supervisors.push(sup);
        streams.push(rx);
    }

    // digests[peer][round] = 16-byte post-ingest det digest.
    let mut digests: Vec<BTreeMap<u64, [u8; 16]>> = vec![BTreeMap::new(); n];
    let last_round = rounds - 1;
    let budget = Duration::from_secs(180 + rounds * (declared_round_timeout_s() + 30));
    let deadline = Instant::now() + budget;
    'collect: loop {
        assert!(
            Instant::now() < deadline,
            "run budget {budget:?} exceeded; rounds so far: {:?}",
            digests.iter().map(BTreeMap::len).collect::<Vec<_>>()
        );
        for (i, rx) in streams.iter_mut().enumerate() {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Event::RoundOutcome { round, digest, .. } => {
                        digests[i].insert(round, digest);
                    }
                    Event::Error { class, detail } => {
                        eprintln!("peer {i} ({}) ERROR {class:?}: {detail}", peers[i].label);
                    }
                    Event::Warning { class, detail } => {
                        eprintln!("peer {i} ({}) WARN [{class}]: {detail}", peers[i].label);
                    }
                    _ => {}
                }
            }
        }
        if (0..n).all(|i| digests[i].contains_key(&last_round)) {
            break 'collect;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Print the digest transcript (the ledger evidence).
    for round in 0..rounds {
        let cols: Vec<String> = (0..n)
            .map(|i| {
                digests[i]
                    .get(&round)
                    .map(hex16)
                    .unwrap_or_else(|| "--".into())
            })
            .collect();
        println!("round {round}: {}", cols.join("  "));
    }

    // The gate assertion: every peer that reported a round agrees byte-for-byte on the det digest.
    for round in 0..rounds {
        let mut reference: Option<(usize, [u8; 16])> = None;
        for (i, d) in digests.iter().enumerate() {
            if let Some(dig) = d.get(&round) {
                match reference {
                    None => reference = Some((i, *dig)),
                    Some((ri, rd)) => assert_eq!(
                        rd, *dig,
                        "round {round}: peer {} ({}) and peer {} ({}) det digests diverge — \
                         cross-platform consensus BROKEN",
                        ri, peers[ri].label, i, peers[i].label
                    ),
                }
            }
        }
    }
    for (i, d) in digests.iter().enumerate() {
        assert_eq!(
            d.len() as u64,
            rounds,
            "peer {} ({}) completed every round",
            i,
            peers[i].label
        );
    }

    // The run reaches Finished.
    let state_url = format!("{}/runs/{run_id}/state", env.ws_base);
    let finish_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (status, state) = get_json(&egress, &env, &state_url).await;
        assert_eq!(status, 200, "GET /state");
        if state["data"]["finished"] == serde_json::Value::Bool(true) {
            println!("final DO state: {}", state["data"]);
            break;
        }
        assert!(
            Instant::now() < finish_deadline,
            "run did not finish: {state}"
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    for sup in &supervisors {
        sup.leave(run_id.clone(), LeaveMode::Immediate).await.ok();
        sup.shutdown().await;
    }
    println!(
        "fleet heterogeneity rehearsal GREEN: {n} peers, {rounds} rounds, det-lane digests byte-identical across platforms"
    );
}

fn hex16(d: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// **Merge-3 P2 WAN-gate ceremony driver — heterogeneous peers + forced churn.**
///
/// The runbook's recommended ceremony driver (§2/§6): the configurable-peer heterogeneity of
/// `fleet_heterogeneous_det_lane_agrees` (each peer a local binary OR a remote `ssh -T` process) PLUS
/// the churn-robust drop→park→rejoin recovery loop of the Merge-2 `ws_live_workers.rs` (which only
/// ever spawned LOCAL peers). `TrainSupervisor` re-execs its configured spawn command on rejoin, so
/// the recovery works transparently for a remote peer (a fresh `ssh` dial) — this is the contained
/// synthesis the lean `fleet_heterogeneous_det_lane_agrees` (no rejoin) could not do, and is why a
/// 4-peer WAN run stalls there but completes here.
///
/// Env: the same `SWARM_FLEET_*` knobs (`author_envelope`/`create_run_request` read them) plus
/// `SWARM_GATE_DROP_INDEX` (peer to kill mid-run; default = last peer) and `SWARM_GATE_DROP_AFTER_ROUND`
/// (default 2). `author_envelope` sets `min_peers = N` so the kill breaches the floor → the run parks
/// in `WaitingForMembers` → the killed peer rejoins (§6.5) → the run resumes and FINISHES.
///
/// Gate assertions (spec §17 / runbook §7): every peer that reports a round agrees on the det digest
/// byte-for-byte (consensus); the kill→park→rejoin cycle is exercised; the run reaches Finished.
///
/// **Lane R (P3) upgrade — the rejoiner is now byte-identical.** With live checkpoint-resync wired
/// (the worker fetches the coordinator's latest checkpoint pointer → `resume_from_checkpoint` →
/// replays the retained rounds), the rejoined peer's POST-rejoin per-round digests are folded back
/// into the main byte-identity assertion (B4's documented exclusion is REMOVED). The gate now asserts
/// "the rejoiner is byte-identical to the survivors after churn", not merely "the run finishes". The
/// drill configures `checkpoint_every_rounds` so a checkpoint exists before the kill.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn fleet_gate_ceremony_with_churn() {
    let Some(env) = fleet_env() else {
        eprintln!(
            "SKIP fleet_gate_ceremony_with_churn: set SWARM_FLEET_WS_URL (+ SWARM_FLEET_PEERS) — see the module docs"
        );
        return;
    };
    let local_bin = ensure_built();
    let module_path = tiny_llama_wasm_path();
    let module_bytes = std::fs::read(&module_path).expect("read tiny_llama.wasm");

    let peers = parse_peers(&local_bin, &module_path);
    let n = peers.len();
    assert!(n >= 2, "need >= 2 peers for a churn ceremony");
    let drop_index = env_usize("SWARM_GATE_DROP_INDEX", n - 1).min(n - 1);
    let drop_after_round: u64 = env_u64("SWARM_GATE_DROP_AFTER_ROUND", 2);
    let global_batch = n as u32 * steps_per_round() * micro_batch();
    let rounds = env.rounds;
    let last_round = rounds - 1;
    println!(
        "GATE CEREMONY: {n} peers {:?}, {rounds} rounds, WS{}, drop peer {} ({}) after round {}",
        peers.iter().map(|p| &p.label).collect::<Vec<_>>(),
        if env.relay_url.is_some() {
            "+iroh"
        } else {
            "-only"
        },
        drop_index,
        peers[drop_index].label,
        drop_after_round,
    );

    let run_id = format!("run-gate-p2-{}", now_secs());
    let envelope = author_envelope(
        &run_id,
        &module_path,
        &module_bytes,
        n as u32,
        rounds,
        global_batch,
    );
    envelope.validate().expect("envelope validates");
    let author = SigningKey::from_bytes(&[0x63u8; 32]);
    let frozen = envelope.freeze(&author).expect("freeze envelope");
    frozen.verify().expect("verify frozen envelope");
    let envelope_hash = frozen.hash().0;
    let wire = to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope");

    let egress = EgressClient::new(EgressConfig::default()).expect("egress client");
    let request = create_run_request(&envelope, &frozen, &module_bytes, rounds, global_batch);
    let (status, text) = post_json(&egress, &env, &format!("{}/runs", env.ws_base), &request).await;
    assert_eq!(status, 201, "POST /runs: {text}");
    println!("created {run_id}");

    // ---- observer tap (the message-log half of the observe capture, carried follow-on 3) --------
    // A passive WsControlPlane subscription records every signed message → `<run>.dsmlog`, verified
    // offline after the run (digest tally + run health). This is the worker-subprocess-loop observe
    // evidence for the ceremony (the P2 gate captured this only via a parallel swarm-local mesh).
    let observe_dir = std::env::var("SWARM_GATE_OBSERVE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("gate-ceremony-observe"));
    let observer = Arc::new(
        WsControlPlane::connect(WsConfig {
            base_url: env.ws_base.clone(),
            run_id: run_id.clone(),
            auth: WsAuth::Internal {
                org_id: env.org.clone(),
                actor: env.actor.clone(),
            },
            reconnect: ReconnectConfig::default(),
        })
        .await
        .expect("observer WS connect"),
    );
    let mut obs_sub = observer.subscribe();
    let (obs_stop_tx, mut obs_stop_rx) = tokio::sync::oneshot::channel::<()>();
    let log_task = {
        let run_id = run_id.clone();
        tokio::spawn(async move {
            let mut log = MessageLog::new(run_id);
            loop {
                tokio::select! {
                    _ = &mut obs_stop_rx => break,
                    m = obs_sub.recv() => match m {
                        Some(bytes) => {
                            if let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) {
                                log.append(msg);
                            }
                        }
                        None => break,
                    },
                }
            }
            log
        })
    };

    // Spawn + assess + join every peer.
    let mut supervisors = Vec::new();
    let mut streams: Vec<Option<tokio::sync::mpsc::UnboundedReceiver<Event>>> = Vec::new();
    let build_sup = |peer: &PeerSpec| {
        let mut cfg = TrainClientConfig::new(&peer.program);
        cfg.args = peer.args.clone();
        if peer.is_local && !push_gate_fetch_env(&mut cfg, &env, "daemon-swarm-cache-gate") {
            cfg.env.push((
                "DAEMON_TRAIN_MODULE".to_string(),
                module_path.to_string_lossy().into_owned(),
            ));
        }
        cfg.spawn_timeout = Duration::from_secs(90);
        cfg.op_timeout = Duration::from_secs(600);
        Arc::new(TrainSupervisor::new(cfg))
    };
    for (i, peer) in peers.iter().enumerate() {
        // Spawn+assess with bounded retry: a remote peer over ssh can transiently fail to spawn
        // (e.g. the Windows box's sshd rate-limits a fresh dial) — a fresh supervisor + short backoff
        // recovers it instead of aborting the whole heterogeneous run at one flaky connection.
        let mut attempt = 0u32;
        let (sup, elig) = loop {
            attempt += 1;
            let sup = build_sup(peer);
            match sup.assess(wire.clone()).await {
                Ok(elig) => break (sup, elig),
                Err(e) if attempt < 4 => {
                    eprintln!(
                        "peer {i} ({}) assess attempt {attempt} failed: {e} — retrying in 6s",
                        peer.label
                    );
                    sup.shutdown().await;
                    tokio::time::sleep(Duration::from_secs(6)).await;
                }
                Err(e) => panic!(
                    "peer {i} ({}) assess (after {attempt} attempts): {e}",
                    peer.label
                ),
            }
        };
        assert!(
            elig.eligible,
            "peer {i} ({}) eligible: {:?}",
            peer.label, elig.reasons
        );
        let creds = credentials_for(i, n, &env, envelope_hash, global_batch)
            .to_bytes()
            .expect("encode credentials");
        let rx = sup
            .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
            .await
            .unwrap_or_else(|e| panic!("peer {i} ({}) join_streaming: {e}", peer.label));
        supervisors.push(sup);
        streams.push(Some(rx));
    }

    // digests[peer][round] = det digests, byte-identity domain. Lane R: the dropped peer's POST-rejoin
    // digests are folded back into `digests[drop_index]` (checkpoint-resync makes them byte-identical
    // to survivors), so they are IN the identity assertion — B4's exclusion is removed.
    let mut digests: Vec<BTreeMap<u64, [u8; 16]>> = vec![BTreeMap::new(); n];
    // Rounds the rejoiner reported AFTER resync (for the transcript + the "resync contributed" check).
    let mut rejoined_rounds: BTreeMap<u64, [u8; 16]> = BTreeMap::new();
    let mut rejoined_stream: Option<tokio::sync::mpsc::UnboundedReceiver<Event>> = None;
    let state_url = format!("{}/runs/{run_id}/state", env.ws_base);

    // Admission gate + diagnostic: with min_peers = N the run cannot leave WaitingForMembers until
    // all N peers attach. A peer whose worker started (assess ok) but whose WS Join never registered
    // (e.g. a bad remote env) would otherwise hang the barrier for the whole budget — so identify it
    // fast. Print each peer's node PeerId, then wait for roster == N (or phase past waiting).
    for (i, peer) in peers.iter().enumerate() {
        let pid = peer_id(&SigningKey::from_bytes(&node_secret(i))).0;
        let hex: String = pid.iter().map(|b| format!("{b:02x}")).collect();
        println!("peer {i} ({}) node_peer_id={hex}", peer.label);
    }
    let admit_deadline = Instant::now() + Duration::from_secs(declared_warmup_s() + 90);
    loop {
        let (_st, state) = get_json(&egress, &env, &state_url).await;
        let roster = state["data"]["roster"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
        let phase = state["data"]["phase"].as_str().unwrap_or("").to_string();
        if roster >= n || phase != "waiting" {
            println!("admission: roster={roster}/{n} phase={phase}");
            break;
        }
        if Instant::now() >= admit_deadline {
            let ids: Vec<String> = state["data"]["roster"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(|s| s.chars().take(16).collect()))
                        .collect()
                })
                .unwrap_or_default();
            panic!(
                "only {roster}/{n} peers admitted after warmup+90s; roster id-prefixes={ids:?} \
                 — a peer's WS Join never registered (match against node_peer_id above)"
            );
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    let budget = Duration::from_secs(300 + rounds * (declared_round_timeout_s() + 60));
    let deadline = Instant::now() + budget;
    let mut dropped = false;
    let mut rejoined = false;
    let mut last_phase = String::new();

    'collect: loop {
        assert!(
            Instant::now() < deadline,
            "run budget {budget:?} exceeded; rounds so far: {:?} (rejoined rounds: {})",
            digests.iter().map(BTreeMap::len).collect::<Vec<_>>(),
            rejoined_rounds.len()
        );
        for (i, slot) in streams.iter_mut().enumerate() {
            if let Some(rx) = slot.as_mut() {
                while let Ok(ev) = rx.try_recv() {
                    match ev {
                        Event::RoundOutcome { round, digest, .. } => {
                            digests[i].insert(round, digest);
                        }
                        Event::Error { class, detail } => {
                            eprintln!("peer {i} ({}) ERROR {class:?}: {detail}", peers[i].label);
                        }
                        Event::Warning { class, detail } => {
                            eprintln!("peer {i} ({}) WARN [{class}]: {detail}", peers[i].label);
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(rx) = rejoined_stream.as_mut() {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Event::RoundOutcome { round, digest, .. } => {
                        // Lane R: fold the rejoiner's post-resync rounds into the identity domain.
                        digests[drop_index].insert(round, digest);
                        rejoined_rounds.insert(round, digest);
                    }
                    Event::ResyncProgress {
                        round,
                        from_checkpoint,
                        replayed,
                        total,
                    } => {
                        eprintln!(
                            "peer {drop_index} ({}) RESYNC replayed round {round} from checkpoint {from_checkpoint} ({replayed}/{total})",
                            peers[drop_index].label
                        );
                    }
                    Event::Warning { class, detail } => {
                        eprintln!(
                            "peer {drop_index} ({}) WARN [{class}]: {detail}",
                            peers[drop_index].label
                        );
                    }
                    _ => {}
                }
            }
        }

        // Kill drill: once the drop target reported `drop_after_round`, kill it.
        if !dropped && digests[drop_index].contains_key(&drop_after_round) {
            println!(
                "CHURN: killing peer {drop_index} ({}) after round {drop_after_round}",
                peers[drop_index].label
            );
            supervisors[drop_index]
                .leave(run_id.clone(), LeaveMode::Immediate)
                .await
                .ok();
            supervisors[drop_index].shutdown().await;
            streams[drop_index] = None;
            dropped = true;
        }

        // Rejoin drill: once the coordinator parks (floor breach → WaitingForMembers), the killed
        // peer's supervisor re-assesses + rejoins (§6.5 previously-Dropped member rejoin). For a
        // remote peer this re-execs a fresh `ssh` dial.
        if dropped && !rejoined {
            let (st, state) = get_json(&egress, &env, &state_url).await;
            assert_eq!(st, 200, "GET /state during churn");
            let ph = format!(
                "{} round={}",
                state["data"]["phase"], state["data"]["round"]
            );
            if ph != last_phase {
                eprintln!("churn-poll: phase={ph}");
                last_phase = ph;
            }
            if state["data"]["phase"] == serde_json::json!("waiting") {
                println!(
                    "CHURN: coordinator parked (floor breach at round {}); rejoining peer {drop_index}",
                    state["data"]["round"]
                );
                let sup = &supervisors[drop_index];
                let elig = sup
                    .assess(wire.clone())
                    .await
                    .expect("re-assess on respawn");
                assert!(elig.eligible, "respawned peer {drop_index} eligible");
                let creds = credentials_for(drop_index, n, &env, envelope_hash, global_batch)
                    .to_bytes()
                    .expect("encode credentials");
                let rx = sup
                    .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
                    .await
                    .expect("respawned peer rejoins");
                rejoined_stream = Some(rx);
                rejoined = true;
            }
        }

        // Lane R: finished when EVERY peer — including the rejoiner (byte-identical via
        // checkpoint-resync) — reported the last round. `min_peers = N` means the run cannot reach
        // the last round without the rejoiner back, so this also proves resync let it re-contribute.
        let all_done = dropped && rejoined && (0..n).all(|i| digests[i].contains_key(&last_round));
        if all_done {
            break 'collect;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(
        dropped && rejoined,
        "the kill→park→rejoin churn cycle was exercised"
    );

    // Transcript (ledger evidence). A `*` marks a round the rejoiner reported POST-resync — folded
    // into the identity domain (lane R), so its column must match the survivors byte-for-byte.
    for round in 0..rounds {
        let cols: Vec<String> = (0..n)
            .map(|i| {
                digests[i]
                    .get(&round)
                    .map(hex16)
                    .unwrap_or_else(|| "--".into())
            })
            .collect();
        let rj = if rejoined_rounds.contains_key(&round) {
            "  [rejoiner resynced ✓]"
        } else {
            ""
        };
        println!("round {round}: {}{rj}", cols.join("  "));
    }

    // Gate assertion (consensus): every peer that reported a round agrees byte-for-byte.
    for round in 0..rounds {
        let mut reference: Option<(usize, [u8; 16])> = None;
        for (i, d) in digests.iter().enumerate() {
            if let Some(dig) = d.get(&round) {
                match reference {
                    None => reference = Some((i, *dig)),
                    Some((ri, rd)) => assert_eq!(
                        rd, *dig,
                        "round {round}: peer {ri} ({}) and peer {i} ({}) det digests diverge — \
                         cross-platform consensus BROKEN",
                        peers[ri].label, peers[i].label
                    ),
                }
            }
        }
    }
    // Survivors completed every round.
    for i in (0..n).filter(|&i| i != drop_index) {
        assert_eq!(
            digests[i].len() as u64,
            rounds,
            "survivor {i} ({}) completed every round",
            peers[i].label
        );
    }
    // Lane R headline: the rejoiner contributed POST-resync rounds (proving it re-attached via
    // checkpoint-resync, not that it merely finished), and — folded into the identity loop above —
    // those digests are byte-identical to the survivors. It reported the last round too.
    assert!(
        !rejoined_rounds.is_empty(),
        "the rejoined peer {drop_index} ({}) contributed at least one post-resync round",
        peers[drop_index].label
    );
    assert!(
        rejoined_rounds.keys().any(|&r| r > drop_after_round),
        "the rejoiner's post-resync rounds are past the drop point (round > {drop_after_round})"
    );
    assert!(
        digests[drop_index].contains_key(&last_round),
        "the rejoined peer {drop_index} ({}) reported the last round byte-identically after resync",
        peers[drop_index].label
    );

    // The run reaches Finished despite the churn.
    let finish_deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let (st, state) = get_json(&egress, &env, &state_url).await;
        assert_eq!(st, 200, "GET /state");
        if state["data"]["finished"] == serde_json::Value::Bool(true) {
            println!("final DO state: {}", state["data"]);
            break;
        }
        assert!(
            Instant::now() < finish_deadline,
            "run did not finish: {state}"
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    // ---- observe capture: stop the tap, write <run>.dsmlog, verify offline ----------------------
    // (Before the worker teardown: a 160M GPU worker can be slow to exit on Leave, and the capture
    // evidence must not depend on teardown promptness.) The rejoiner leaves a gap while parked, so a
    // round's reporter count over the whole log can be < n during the churn window; assert agreement
    // among the reporters that DID report (never a divergence), which is the consensus bar.
    let _ = obs_stop_tx.send(());
    observer.shutdown().await;
    let log = log_task.await.expect("observer log task");
    std::fs::create_dir_all(&observe_dir).expect("create observe dir");
    let log_path = observe_dir.join(format!("{run_id}.dsmlog"));
    let mut file = std::fs::File::create(&log_path).expect("create dsmlog");
    log.write_to(&mut file).expect("write dsmlog");
    println!(
        "observe: wrote {} ({} messages, rounds {:?})",
        log_path.display(),
        log.len(),
        log.rounds()
    );
    let mut reader = std::fs::File::open(&log_path).expect("open dsmlog");
    let reread = MessageLog::read_from(&mut reader).expect("read dsmlog back");
    assert_eq!(reread.len(), log.len(), "dsmlog round-trips");
    let health = RunHealth::from_log(&reread);
    for round in 0..rounds {
        let verdict = digest_tally_from_log(&reread, round, n as u32);
        println!(
            "observe: round {round} digest tally — reporters={} agreed={} outliers={:?}",
            verdict.reporters, verdict.agreed, verdict.outliers
        );
        assert!(
            verdict.agreed,
            "round {round}: offline digest tally disagreed (outliers {:?}) — consensus BROKEN",
            verdict.outliers
        );
    }
    println!(
        "observe: run health — {} rounds projected from the log",
        health.rounds.len()
    );

    for sup in &supervisors {
        let _ = tokio::time::timeout(
            Duration::from_secs(30),
            sup.leave(run_id.clone(), LeaveMode::Immediate),
        )
        .await;
        let _ = tokio::time::timeout(Duration::from_secs(30), sup.shutdown()).await;
    }
    println!(
        "GATE CEREMONY GREEN: {n} heterogeneous peers, {rounds} rounds, det digests byte-identical \
         (incl. the rejoiner), churn (kill peer {drop_index} → park → checkpoint-resync → rejoin) \
         survived, run Finished. rejoiner reported {} post-resync round(s) BYTE-IDENTICAL to \
         survivors: {:?}. observe log: {}",
        rejoined_rounds.len(),
        rejoined_rounds.keys().collect::<Vec<_>>(),
        log_path.display(),
    );
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
}

/// Run `n` LOCAL worker peers through `rounds` rounds against the live coordinator, returning the
/// per-round *inter-round wall* (ms): the steady-state time between consecutive rounds becoming
/// complete-for-all-peers. The first delta already excludes warmup (warmup precedes round-0
/// completion). Used by the overhead measurement — the single-host baseline (`n=1`) vs the multi-peer
/// swarm — so `overhead% = (mean(t_N) - mean(t_1)) / mean(t_1)`, i.e. the round overhead vs a
/// single-host reference (spec §17: round overhead incl. the §6.4 barrier ingest gap).
async fn timed_local_run(
    env: &FleetEnv,
    local_bin: &Path,
    module_path: &Path,
    module_bytes: &[u8],
    n: usize,
    rounds: u64,
    tag: &str,
) -> Vec<f64> {
    let global_batch = n as u32 * steps_per_round() * micro_batch();
    let run_id = format!("run-c3-oh-{tag}-{}", now_secs());
    let envelope = author_envelope(
        &run_id,
        module_path,
        module_bytes,
        n as u32,
        rounds,
        global_batch,
    );
    envelope.validate().expect("envelope validates");
    let author = SigningKey::from_bytes(&[0xC4u8; 32]);
    let frozen = envelope.freeze(&author).expect("freeze");
    let envelope_hash = frozen.hash().0;
    let wire = to_canonical_vec(&frozen.to_wire()).expect("encode envelope");
    let egress = EgressClient::new(EgressConfig::default()).expect("egress");
    let request = create_run_request(&envelope, &frozen, module_bytes, rounds, global_batch);
    let (status, text) = post_json(&egress, env, &format!("{}/runs", env.ws_base), &request).await;
    assert_eq!(status, 201, "POST /runs ({tag}): {text}");

    let mut supervisors = Vec::new();
    let mut streams = Vec::new();
    for i in 0..n {
        let mut cfg = TrainClientConfig::new(local_bin);
        if !push_gate_fetch_env(&mut cfg, env, "daemon-swarm-cache-oh") {
            cfg.env.push((
                "DAEMON_TRAIN_MODULE".to_string(),
                module_path.to_string_lossy().into_owned(),
            ));
        }
        cfg.spawn_timeout = Duration::from_secs(90);
        cfg.op_timeout = Duration::from_secs(600);
        let sup = Arc::new(TrainSupervisor::new(cfg));
        let elig = sup.assess(wire.clone()).await.expect("assess");
        assert!(elig.eligible, "peer {i} eligible ({tag})");
        let creds = credentials_for(i, n, env, envelope_hash, global_batch)
            .to_bytes()
            .expect("creds");
        let rx = sup
            .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
            .await
            .expect("join");
        supervisors.push(sup);
        streams.push(rx);
    }

    let start = Instant::now();
    let mut round_complete_ms: BTreeMap<u64, f64> = BTreeMap::new();
    let mut seen: Vec<BTreeMap<u64, ()>> = vec![BTreeMap::new(); n];
    let last = rounds - 1;
    let deadline =
        Instant::now() + Duration::from_secs(180 + rounds * (declared_round_timeout_s() + 30));
    loop {
        assert!(
            Instant::now() < deadline,
            "overhead {tag} run stalled: {round_complete_ms:?}"
        );
        for (i, rx) in streams.iter_mut().enumerate() {
            while let Ok(ev) = rx.try_recv() {
                if let Event::RoundOutcome { round, .. } = ev {
                    seen[i].insert(round, ());
                }
            }
        }
        for r in 0..rounds {
            if !round_complete_ms.contains_key(&r) && (0..n).all(|i| seen[i].contains_key(&r)) {
                round_complete_ms.insert(r, start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        if round_complete_ms.contains_key(&last) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    for sup in &supervisors {
        sup.leave(run_id.clone(), LeaveMode::Immediate).await.ok();
        sup.shutdown().await;
    }
    let mut walls = Vec::new();
    let mut prev: Option<f64> = None;
    for r in 0..rounds {
        if let Some(t) = round_complete_ms.get(&r) {
            if let Some(p) = prev {
                walls.push(t - p);
            }
            prev = Some(*t);
        }
    }
    walls
}

/// **C3 overhead measurement (gate criterion, spec §17: round overhead <15%).** Runs the SAME real
/// workload (tiny-llama guest, real R2 payload plane) as a single-host baseline (1 peer) and as a
/// multi-peer swarm (N peers, `SWARM_FLEET_OVERHEAD_PEERS`, default 4), all LOCAL subprocesses, and
/// reports `overhead% = (mean N-peer round wall - mean single-host round wall) / single-host`.
///
/// NB (honest caveat): the tiny-llama guest's per-round COMPUTE is sub-millisecond, so this ratio is
/// protocol/latency-dominated and is NOT the gate-representative figure — it validates the
/// *measurement mechanism* + the barrier scaling. The gate's real <15% comes from running this exact
/// tool with the 160M preset (seconds of compute per round), where the barrier is a small fraction.
/// See swarm-p2-gate-runbook.md §overhead for the full procedure + the single-host 160M baseline
/// (throughput harness, swarm-p2-throughput.md).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn swarm_round_overhead_vs_single_host() {
    let Some(env) = fleet_env() else {
        eprintln!("SKIP swarm_round_overhead: set SWARM_FLEET_WS_URL");
        return;
    };
    let local_bin = ensure_built();
    let module_path = tiny_llama_wasm_path();
    let module_bytes = std::fs::read(&module_path).expect("read module");
    let rounds = env.rounds.max(5);
    let n = std::env::var("SWARM_FLEET_OVERHEAD_PEERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize);

    let baseline = timed_local_run(
        &env,
        &local_bin,
        &module_path,
        &module_bytes,
        1,
        rounds,
        "base",
    )
    .await;
    let swarm = timed_local_run(
        &env,
        &local_bin,
        &module_path,
        &module_bytes,
        n,
        rounds,
        "swarm",
    )
    .await;

    let mean = |v: &[f64]| -> f64 {
        // Drop the first steady-state delta (round0->1) as a further warmup guard when we have >=3.
        let s: &[f64] = if v.len() >= 3 { &v[1..] } else { v };
        if s.is_empty() {
            0.0
        } else {
            s.iter().sum::<f64>() / s.len() as f64
        }
    };
    let t1 = mean(&baseline);
    let tn = mean(&swarm);
    let overhead = if t1 > 0.0 {
        (tn - t1) / t1 * 100.0
    } else {
        f64::NAN
    };
    println!("=== C3 round-overhead measurement (tiny-llama scale; mechanism validation) ===");
    println!("single-host baseline (1 peer)  mean round wall t1 = {t1:.1} ms");
    println!("{n}-peer swarm                   mean round wall tN = {tn:.1} ms");
    println!("round overhead vs single host   = {overhead:.1} %  (tiny-llama: protocol-dominated, NOT the gate figure)");
    println!("  baseline per-round walls (ms): {baseline:?}");
    println!("  {n}-peer  per-round walls (ms): {swarm:?}");
    assert!(t1 > 0.0 && tn > 0.0, "measured both round walls");
}
