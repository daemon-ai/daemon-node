// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **P3 lane S — the 160M staging rehearsal (the Merge-2 pre-staging exit criterion).**
//!
//! The P2 gate pre-staged the experiment `.wasm` on every fleet box (`DAEMON_TRAIN_MODULE`) and used
//! a synthetic corpus. This harness proves the replacement end to end: N local `daemon-train-worker`
//! subprocesses join a real live run and fetch **everything** from the payload store by content hash
//! — the module (`modules/<blake3>.wasm`, resolved at assess via the envelope artifact map) and the
//! pre-tokenized corpus (`corpus/<blake3>.{json,bin}`, resolved at join via `EngineParams.corpus`) —
//! blake3-verified before use, cached content-addressed on disk, with **no local module path and no
//! pre-staged bytes anywhere**. Rounds train the real 160M preset (Vulkan via
//! `DAEMON_TRAIN_BACKEND=wgpu` on this box) and the det digests must be byte-identical.
//!
//! An **observer tap** subscribes a passive `WsControlPlane` to the run and writes the signed-message
//! log as `<run>.dsmlog` (the observe surface's message-log artifact), then verifies it offline
//! (`RunHealth::from_log` + per-round digest tally) — the worker-subprocess capture the P2 gate
//! lacked (carried follow-on 3, message-log half; the engine-input `.dsmcap` half still rides the
//! `swarm-local` harness path).
//!
//! Env-gated (skips unless configured). Publish the artifacts first (once — content-addressed):
//! ```text
//! cargo run -p xtask -- publish-module --module guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm \
//!   --run <scope> --presign-base <base> --org org_live --actor key:live
//! cargo run -p xtask -- tokenize-corpus --dataset roneneldan/TinyStories --dataset-file TinyStories-valid.txt \
//!   --tokenizer gpt2 --out-dir /tmp/ts-corpus --seq-len 1024 --max-tokens 2000000
//! cargo run -p xtask -- publish-corpus --manifest /tmp/ts-corpus/manifest.json --run <scope> …
//! ```
//! Then:
//! ```text
//! SWARM_STAGE_WS_URL=https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm \
//! SWARM_STAGE_ASSET_SCOPE=<scope> \
//! SWARM_STAGE_MODULE_BLAKE3=<hex> SWARM_STAGE_MODULE_SIZE=<bytes> \
//! SWARM_STAGE_MANIFEST_BLAKE3=<hex> SWARM_STAGE_MANIFEST_SIZE=<bytes> \
//! SWARM_STAGE_ROUNDS=3 SWARM_STAGE_BACKEND=wgpu \
//!   cargo test -p daemon-swarm-e2e --features iroh --test staging_160m -- --nocapture --test-threads 1
//! ```
//! Knobs: `SWARM_STAGE_PEERS` (default 2), `SWARM_STAGE_REDUCED=1` (reduced-preset smoke instead of
//! the full 160M), `SWARM_STAGE_WORKER_FEATURES` (default `swarm-net,wgpu`), `SWARM_STAGE_OBSERVE_DIR`
//! (default `/tmp/stage-160m-observe`), `SWARM_STAGE_{WARMUP,ROUND_TIMEOUT,COOLDOWN}_S`.
#![cfg(feature = "iroh")]
// Test harness: builds the worker via cargo, reads operator-local files, prints progress (the
// sanctioned dev-tool exception, mirroring ws_live_workers.rs / fleet_live_hetero.rs).
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_swarm_net::ControlPlane;
use daemon_swarm_net::{ReconnectConfig, WsAuth, WsConfig, WsControlPlane};
use daemon_swarm_observe::desync::digest_tally_from_log;
use daemon_swarm_observe::{MessageLog, RunHealth};
use daemon_swarm_run::protocol::{
    CorpusRef, EngineParams, Event, JoinCredentials, JoinPolicy, LeaveMode, PolicyMode, WsAuthSpec,
};
use daemon_train_client::{TrainClientConfig, TrainSupervisor};
use daemon_train_sdk::models::TinyLlamaCfg;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    from_canonical_slice, peer_id, to_canonical_vec, SignedMessage, SigningKey,
};

const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 1;

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

struct StageEnv {
    ws_base: String,
    presign_base: String,
    asset_scope: String,
    org: String,
    actor: String,
    rounds: u64,
    peers: usize,
    module_blake3: [u8; 32],
    module_size: u64,
    manifest_blake3: [u8; 32],
    manifest_size: u64,
    backend: String,
    observe_dir: PathBuf,
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

fn stage_env() -> Option<StageEnv> {
    let ws_base = std::env::var("SWARM_STAGE_WS_URL").ok()?;
    let module_blake3 = hex32(&std::env::var("SWARM_STAGE_MODULE_BLAKE3").ok()?)?;
    let manifest_blake3 = hex32(&std::env::var("SWARM_STAGE_MANIFEST_BLAKE3").ok()?)?;
    Some(StageEnv {
        presign_base: std::env::var("SWARM_STAGE_PRESIGN_BASE").unwrap_or_else(|_| ws_base.clone()),
        asset_scope: std::env::var("SWARM_STAGE_ASSET_SCOPE")
            .unwrap_or_else(|_| "assets-p3s".into()),
        org: std::env::var("SWARM_STAGE_ORG").unwrap_or_else(|_| "org_live".into()),
        actor: std::env::var("SWARM_STAGE_ACTOR").unwrap_or_else(|_| "key:live".into()),
        rounds: env_u64("SWARM_STAGE_ROUNDS", 3),
        peers: env_u64("SWARM_STAGE_PEERS", 2) as usize,
        module_blake3,
        module_size: env_u64("SWARM_STAGE_MODULE_SIZE", 0),
        manifest_blake3,
        manifest_size: env_u64("SWARM_STAGE_MANIFEST_SIZE", 0),
        backend: std::env::var("SWARM_STAGE_BACKEND").unwrap_or_else(|_| "wgpu".into()),
        observe_dir: std::env::var("SWARM_STAGE_OBSERVE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/stage-160m-observe")),
        ws_base,
    })
}

fn declared_warmup_s() -> u64 {
    env_u64("SWARM_STAGE_WARMUP_S", 120)
}
fn declared_round_timeout_s() -> u64 {
    env_u64("SWARM_STAGE_ROUND_TIMEOUT_S", 300)
}
fn declared_cooldown_s() -> u64 {
    env_u64("SWARM_STAGE_COOLDOWN_S", 3)
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

static BUILD: Once = Once::new();

/// Build the worker with the staging feature set (default `swarm-net,wgpu` — the Vulkan lane).
fn ensure_built() -> PathBuf {
    let features = std::env::var("SWARM_STAGE_WORKER_FEATURES")
        .unwrap_or_else(|_| "swarm-net,wgpu".to_string());
    BUILD.call_once(|| {
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

/// The 160M preset config (or the reduced smoke variant under `SWARM_STAGE_REDUCED=1`).
fn stage_cfg() -> TinyLlamaCfg {
    if std::env::var_os("SWARM_STAGE_REDUCED").is_some() {
        TinyLlamaCfg {
            n_layers: 1,
            d_model: 256,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 64,
            vocab: 50257, // GPT-2 BPE — matches the published TinyStories shards
            seq_len: 1024,
            ..TinyLlamaCfg::llama_160m()
        }
    } else {
        TinyLlamaCfg::llama_160m()
    }
}

fn cfg_value(cfg: &TinyLlamaCfg) -> ciborium::value::Value {
    ciborium::value::Value::serialized(cfg).expect("preset config serializes")
}

fn author_envelope(env: &StageEnv, run_id: &str, peers: u32, global_batch: u32) -> Envelope {
    let module_hex: String = env
        .module_blake3
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let manifest_hex: String = env
        .manifest_blake3
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut artifacts = std::collections::BTreeMap::new();
    // The content-addressed module key (P3 lane S): the worker fetches + blake3-verifies it from
    // the store — no DAEMON_TRAIN_MODULE anywhere.
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: format!("r2://modules/{module_hex}.wasm"),
            blake3: daemon_vhc_proto::Hash::new(env.module_blake3),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: format!("r2://corpus/{manifest_hex}.json"),
            blake3: daemon_vhc_proto::Hash::new(env.manifest_blake3),
        },
    );
    Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: run_id.to_string(),
            min_peers: peers,
            max_peers: peers + 1,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".to_string(),
            abi: "tabi@1".to_string(),
            config: cfg_value(&stage_cfg()),
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".to_string(),
            steps_per_round: STEPS_PER_ROUND,
            global_batch: GlobalBatch {
                start: global_batch,
                end: global_batch,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(env.rounds),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 0,
            uplink_mbps_min: 0,
            downlink_mbps_min: 0,
            disk_gb_min: 0,
            throughput_floor: "c1".to_string(),
            update_mb_max: 64,
            capabilities: Vec::new(),
            payload_store: "r2".to_string(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: declared_warmup_s() as u32,
            round_train_max: declared_round_timeout_s() as u32,
            round_witness: 60,
            cooldown: declared_cooldown_s() as u32,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 3,
            payload_retention_rounds: 16,
        },
    }
}

async fn post_json(
    egress: &EgressClient,
    env: &StageEnv,
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

async fn get_json(egress: &EgressClient, env: &StageEnv, url: &str) -> (u16, serde_json::Value) {
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
    env: &StageEnv,
    envelope: &Envelope,
    frozen: &daemon_vhc_proto::FrozenEnvelope,
    global_batch: u32,
) -> serde_json::Value {
    use base64::Engine as _;
    let module_hex: String = env
        .module_blake3
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let manifest_hex: String = env
        .manifest_blake3
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    serde_json::json!({
        "run_id": envelope.run.run_id,
        "schema": ENVELOPE_SCHEMA_MAJOR,
        "proto_version": daemon_vhc_proto::SWARM_PROTO_VERSION,
        "envelope_b64": base64::engine::general_purpose::STANDARD.encode(frozen.bytes()),
        "author_pubkey": frozen.signer().to_hex(),
        "signature": frozen.signature().to_hex(),
        "artifacts": [
            { "path": format!("modules/{module_hex}.wasm"), "blake3": module_hex, "size": env.module_size },
            { "path": format!("corpus/{manifest_hex}.json"), "blake3": manifest_hex, "size": env.manifest_size },
        ],
        "update_max_bytes": u64::from(envelope.requirements.update_mb_max) * 1024 * 1024,
        "min_peers": envelope.run.min_peers,
        "max_peers": envelope.run.max_peers,
        "rounds": env.rounds,
        "warmup_timeout_s": declared_warmup_s(),
        "round_timeout_s": declared_round_timeout_s(),
        "cooldown_s": declared_cooldown_s(),
        "global_batch": global_batch,
        "witness_target": 0,
    })
}

fn node_secret(i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0xA1 + i as u8;
    s[1] = 0x60;
    s
}

fn credentials_for(
    i: usize,
    n: usize,
    env: &StageEnv,
    envelope_hash: [u8; 32],
    window_sequences: u64,
) -> JoinCredentials {
    let roster: Vec<[u8; 32]> = (0..n)
        .map(|j| peer_id(&SigningKey::from_bytes(&node_secret(j))).0)
        .collect();
    JoinCredentials {
        node_secret: node_secret(i),
        ws_auth: WsAuthSpec::Internal {
            org_id: env.org.clone(),
            actor: env.actor.clone(),
        },
        roster,
        envelope_hash,
        iroh: None, // WS-only: the T0 baseline suffices for the staging rehearsal
        presign_base: Some(env.presign_base.clone()),
        engine: EngineParams {
            steps_per_round: STEPS_PER_ROUND,
            micro_batch: MICRO_BATCH,
            stall_rounds_max: 3,
            checkpoint_every_rounds: 0,
            update_max_bytes: 64 << 20,
            // Merge-2: R's §9 resync-replay window (additive; the staging rehearsal keeps no
            // checkpoints so this is inert here, but the field is required by the merged struct).
            payload_retention_rounds: 16,
            // Synthetic fallback fields — UNUSED when `corpus` is Some (kept for the wire shape).
            corpus_seed: 0,
            corpus_shards: 0,
            corpus_tokens_per_shard: 0,
            corpus_seq_len: 0,
            corpus_vocab_clamp: 0, // real GPT-2 BPE tokens are already inside vocab 50257
            corpus: Some(CorpusRef {
                manifest_blake3: env.manifest_blake3,
                manifest_size: env.manifest_size,
                window_start: 0,
                window_sequences,
            }),
        },
    }
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

fn hex16(d: &[u8; 16]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn staging_160m_fetch_by_hash_rehearsal() {
    let Some(env) = stage_env() else {
        eprintln!(
            "SKIP staging_160m: set SWARM_STAGE_WS_URL + SWARM_STAGE_MODULE_BLAKE3 + \
             SWARM_STAGE_MANIFEST_BLAKE3 (see the module docs)"
        );
        return;
    };
    let local_bin = ensure_built();
    let n = env.peers.max(2);
    let global_batch = n as u32 * STEPS_PER_ROUND * MICRO_BATCH;
    let window_sequences = env.rounds * u64::from(global_batch);
    let run_id = format!(
        "run-s-160m-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs()
    );
    println!(
        "160M staging rehearsal: {n} peers, {} rounds, backend={}, run={run_id}, asset scope={}",
        env.rounds, env.backend, env.asset_scope
    );

    let envelope = author_envelope(&env, &run_id, n as u32, global_batch);
    envelope.validate().expect("envelope validates");
    let author = SigningKey::from_bytes(&[0x51u8; 32]);
    let frozen = envelope.freeze(&author).expect("freeze envelope");
    frozen.verify().expect("verify frozen envelope");
    let envelope_hash = frozen.hash().0;
    let wire = to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope");

    let egress = EgressClient::new(EgressConfig::default()).expect("egress client");
    let request = create_run_request(&env, &envelope, &frozen, global_batch);
    let (status, text) = post_json(&egress, &env, &format!("{}/runs", env.ws_base), &request).await;
    assert_eq!(status, 201, "POST /runs: {text}");
    println!("created {run_id}");

    // ---- observer tap (the message-log half of the observe capture, carried follow-on 3) --------
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
    // The tap stops on an explicit signal: the subscription's sender lives on the plane handle (not
    // its task), so `recv()` alone would never observe a close while the observer Arc is alive.
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

    // ---- spawn + assess + join the workers (NO DAEMON_TRAIN_MODULE — fetch-by-hash only) --------
    let cache_dir = std::env::temp_dir().join("daemon-swarm-cache-stage160m");
    let mut supervisors = Vec::new();
    let mut streams = Vec::new();
    for i in 0..n {
        let mut cfg = TrainClientConfig::new(&local_bin);
        // The store-fetch context (small env strings): presign target + shared asset scope + the
        // on-disk content cache. The module/corpus BYTES travel only through the store.
        cfg.env.push((
            "DAEMON_SWARM_PRESIGN_BASE".to_string(),
            env.presign_base.clone(),
        ));
        cfg.env
            .push(("DAEMON_SWARM_RUN_ID".to_string(), env.asset_scope.clone()));
        cfg.env
            .push(("DAEMON_SWARM_ORG".to_string(), env.org.clone()));
        cfg.env
            .push(("DAEMON_SWARM_ACTOR".to_string(), env.actor.clone()));
        cfg.env.push((
            "DAEMON_SWARM_CACHE_DIR".to_string(),
            cache_dir.to_string_lossy().into_owned(),
        ));
        cfg.env
            .push(("DAEMON_TRAIN_BACKEND".to_string(), env.backend.clone()));
        cfg.spawn_timeout = Duration::from_secs(90);
        cfg.op_timeout = Duration::from_secs(600); // 160M assess/meta + module fetch can take a while
        let sup = Arc::new(TrainSupervisor::new(cfg));
        let t0 = Instant::now();
        let elig = sup
            .assess(wire.clone())
            .await
            .unwrap_or_else(|e| panic!("peer {i} assess (module fetch-by-hash): {e}"));
        println!(
            "peer {i}: assess ok in {:.1}s (module fetched by hash + verified) eligible={} {:?}",
            t0.elapsed().as_secs_f64(),
            elig.eligible,
            elig.reasons
        );
        assert!(elig.eligible, "peer {i} eligible: {:?}", elig.reasons);
        let creds = credentials_for(i, n, &env, envelope_hash, window_sequences)
            .to_bytes()
            .expect("encode credentials");
        let rx = sup
            .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
            .await
            .unwrap_or_else(|e| panic!("peer {i} join_streaming: {e}"));
        supervisors.push(sup);
        streams.push(rx);
    }

    // ---- collect per-round det digests ----------------------------------------------------------
    let mut digests: Vec<BTreeMap<u64, [u8; 16]>> = vec![BTreeMap::new(); n];
    let last_round = env.rounds - 1;
    let budget = Duration::from_secs(
        declared_warmup_s() + 120 + env.rounds * (declared_round_timeout_s() + 60),
    );
    let deadline = Instant::now() + budget;
    loop {
        assert!(
            Instant::now() < deadline,
            "run budget {budget:?} exceeded; rounds so far: {:?}",
            digests.iter().map(BTreeMap::len).collect::<Vec<_>>()
        );
        for (i, rx) in streams.iter_mut().enumerate() {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Event::RoundOutcome { round, digest, .. } => {
                        println!("peer {i}: round {round} digest {}", hex16(&digest));
                        digests[i].insert(round, digest);
                    }
                    Event::Metric { name, value } if name == "loss" => {
                        println!("peer {i}: loss {value:.4}");
                    }
                    Event::Error { class, detail } => {
                        eprintln!("peer {i} ERROR {class:?}: {detail}");
                    }
                    Event::Warning { class, detail } => {
                        eprintln!("peer {i} WARN [{class}]: {detail}");
                    }
                    _ => {}
                }
            }
        }
        if (0..n).all(|i| digests[i].contains_key(&last_round)) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Transcript + byte-identity (the consensus bar).
    for round in 0..env.rounds {
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
    for round in 0..env.rounds {
        let mut reference: Option<[u8; 16]> = None;
        for (i, d) in digests.iter().enumerate() {
            if let Some(dig) = d.get(&round) {
                match reference {
                    None => reference = Some(*dig),
                    Some(rd) => assert_eq!(
                        rd, *dig,
                        "round {round}: peer {i} det digest diverges — consensus BROKEN"
                    ),
                }
            }
        }
    }

    // The run reaches Finished.
    let state_url = format!("{}/runs/{run_id}/state", env.ws_base);
    let finish_deadline = Instant::now() + Duration::from_secs(90);
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

    // ---- observe capture: stop the tap, write <run>.dsmlog, verify offline ----------------------
    // (BEFORE the worker teardown: a 160M wgpu worker can be slow to exit on Leave, and the capture
    // evidence must not depend on teardown promptness.)
    let _ = obs_stop_tx.send(());
    observer.shutdown().await;
    let log = log_task.await.expect("observer log task");
    std::fs::create_dir_all(&env.observe_dir).expect("create observe dir");
    let log_path = env.observe_dir.join(format!("{run_id}.dsmlog"));
    let mut file = std::fs::File::create(&log_path).expect("create dsmlog");
    log.write_to(&mut file).expect("write dsmlog");
    println!(
        "observe: wrote {} ({} messages, rounds {:?})",
        log_path.display(),
        log.len(),
        log.rounds()
    );
    // Offline verification from the written artifact alone (read back, tally digests, run health).
    let mut reader = std::fs::File::open(&log_path).expect("open dsmlog");
    let reread = MessageLog::read_from(&mut reader).expect("read dsmlog back");
    assert_eq!(reread.len(), log.len(), "dsmlog round-trips");
    let health = RunHealth::from_log(&reread);
    for round in 0..env.rounds {
        let verdict = digest_tally_from_log(&reread, round, n as u32);
        println!(
            "observe: round {round} digest tally — reporters={} agreed={} outliers={:?}",
            verdict.reporters, verdict.agreed, verdict.outliers
        );
        assert!(
            verdict.agreed && verdict.reporters >= n as u32,
            "round {round}: offline digest tally must agree across all {n} reporters"
        );
    }
    println!(
        "observe: run health — {} rounds projected from the log",
        health.rounds.len()
    );

    println!(
        "160M STAGING REHEARSAL GREEN: {n} peers fetched module+corpus purely by content hash \
         (no pre-staging), {} rounds, det digests byte-identical, observe log captured + verified",
        env.rounds
    );

    // Teardown last, bounded: a 160M wgpu worker can be slow to service Leave (its engine task owns
    // the device). The assertions above are already complete; don't let teardown hang the harness.
    for sup in &supervisors {
        let _ = tokio::time::timeout(
            Duration::from_secs(30),
            sup.leave(run_id.clone(), LeaveMode::Immediate),
        )
        .await;
        let _ = tokio::time::timeout(Duration::from_secs(30), sup.shutdown()).await;
    }
}
