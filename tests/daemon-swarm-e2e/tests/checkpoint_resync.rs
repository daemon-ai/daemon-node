// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
//! **Lane R (P3) focused live checkpoint-resync drill.**
//!
//! The headline exit criterion: a peer that drops mid-run and respawns rejoins with state
//! **byte-identical to the survivors** — because the worker now (a) reads the coordinator's latest
//! published checkpoint pointer (`GET /state.checkpoint`), (b) `resume_from_checkpoint`s it, and
//! (c) replays the retained rounds forward (`resync_by_replay`) before contributing again (spec §9).
//! This is the property B4 proved in-process (`run_units.rs`) and A3 left as a design note, now wired
//! into the LIVE cloud-DO worker loop.
//!
//! Two tests, both against wrangler-dev (`SWARM_LIVE_WS_URL`; skips cleanly when unset):
//! - `resync_rejoiner_is_byte_identical` — checkpoint cadence on, kill→park→respawn→resync→rejoin;
//!   the rejoiner's POST-rejoin per-round digests are asserted byte-identical to survivors.
//! - `fresh_state_rejoin_still_finishes` — checkpoint cadence OFF (no checkpoint published), same
//!   churn; the rejoiner falls back to fresh-state and the run still FINISHES (the §9 first-epoch /
//!   no-checkpoint honest edge — the pre-P3 behavior is preserved).
//!
//! Drive it (a wrangler-dev on 8795, this branch's coordinator):
//! ```text
//! SWARM_LIVE_WS_URL=http://127.0.0.1:8795/api/v1/swarm \
//!   cargo test -p daemon-swarm-e2e --features iroh --test checkpoint_resync -- --nocapture --test-threads 1
//! ```

// Test harness: builds the worker binary + guest via cargo (the sanctioned dev-tool exception,
// mirroring ws_live_workers.rs), spawns local subprocesses, and prints progress.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_swarm_run::protocol::{
    EngineParams, Event, JoinCredentials, JoinPolicy, LeaveMode, PolicyMode, WsAuthSpec,
};
use daemon_train_client::{TrainClientConfig, TrainSupervisor};
use daemon_train_sdk::models::TinyLlamaCfg;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{peer_id, to_canonical_vec, SigningKey};

const NUM_WORKERS: usize = 3;
const ROUNDS: u64 = 8;
const GUEST_VOCAB: u32 = 64;
const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 2;
const WARMUP_S: u32 = 8;
const ROUND_TIMEOUT_S: u32 = 20;
const COOLDOWN_S: u32 = 1;
const DROP_AFTER_ROUND: u64 = 2;

struct LiveEnv {
    ws_base: String,
    presign_base: String,
    org: String,
    actor: String,
}

fn live_env() -> Option<LiveEnv> {
    let ws_base = std::env::var("SWARM_LIVE_WS_URL").ok()?;
    Some(LiveEnv {
        presign_base: std::env::var("SWARM_LIVE_PRESIGN_BASE").unwrap_or_else(|_| ws_base.clone()),
        org: std::env::var("SWARM_LIVE_ORG").unwrap_or_else(|_| "org_live".into()),
        actor: std::env::var("SWARM_LIVE_ACTOR").unwrap_or_else(|_| "key:live".into()),
        ws_base,
    })
}

// ---- worker binary + guest build (mirrors ws_live_workers.rs) ------------------------------------

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    workspace_root().join("guests/target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

fn ensure_built() -> PathBuf {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(workspace_root())
            .args([
                "build",
                "-p",
                "daemon-train",
                "--features",
                "swarm-net",
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
    global_batch: u32,
) -> Envelope {
    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        "tiny_llama.wasm".to_string(),
        Artifact {
            url: format!("file://{}", module_path.display()),
            blake3: daemon_vhc_proto::blake3_hash(module_bytes),
        },
    );
    Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: run_id.to_string(),
            min_peers: NUM_WORKERS as u32,
            max_peers: NUM_WORKERS as u32 + 1,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "tiny_llama.wasm".to_string(),
            abi: "tabi@1".to_string(),
            config: tiny_cfg(),
        },
        artifacts,
        data: DataSection {
            manifest: "tiny_llama.wasm".to_string(),
            steps_per_round: STEPS_PER_ROUND,
            global_batch: GlobalBatch {
                start: global_batch,
                end: global_batch,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(ROUNDS),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 0,
            uplink_mbps_min: 0,
            downlink_mbps_min: 0,
            disk_gb_min: 0,
            throughput_floor: "c1".to_string(),
            update_mb_max: 1,
            capabilities: Vec::new(),
            payload_store: "r2".to_string(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: WARMUP_S,
            round_train_max: ROUND_TIMEOUT_S,
            round_witness: 30,
            cooldown: COOLDOWN_S,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 3,
            payload_retention_rounds: 16,
        },
    }
}

async fn post_json(
    egress: &EgressClient,
    env: &LiveEnv,
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

async fn get_json(egress: &EgressClient, env: &LiveEnv, url: &str) -> (u16, serde_json::Value) {
    let req = EgressRequest::get(url)
        .header("x-daemon-org-id", &env.org)
        .header("x-daemon-actor", &env.actor);
    let resp = egress
        .execute(req, Redirects::None)
        .await
        .expect("registry GET");
    let status = resp.status().as_u16();
    let body = resp.bytes().await.expect("read body");
    (
        status,
        serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null),
    )
}

fn create_run_request(
    envelope: &Envelope,
    frozen: &daemon_vhc_proto::FrozenEnvelope,
    module_bytes: &[u8],
    global_batch: u32,
) -> serde_json::Value {
    use base64::Engine as _;
    serde_json::json!({
        "run_id": envelope.run.run_id,
        "schema": ENVELOPE_SCHEMA_MAJOR,
        "proto_version": daemon_vhc_proto::SWARM_PROTO_VERSION,
        "envelope_b64": base64::engine::general_purpose::STANDARD.encode(frozen.bytes()),
        "author_pubkey": frozen.signer().to_hex(),
        "signature": frozen.signature().to_hex(),
        "artifacts": [{
            "path": "tiny_llama.wasm",
            "blake3": daemon_vhc_proto::blake3_hash(module_bytes).to_hex(),
            "size": module_bytes.len(),
        }],
        "update_max_bytes": u64::from(envelope.requirements.update_mb_max) * 1024 * 1024,
        "min_peers": envelope.run.min_peers,
        "max_peers": envelope.run.max_peers,
        "rounds": ROUNDS,
        "warmup_timeout_s": WARMUP_S,
        "round_timeout_s": ROUND_TIMEOUT_S,
        "cooldown_s": COOLDOWN_S,
        "global_batch": global_batch,
        "witness_target": 0,
    })
}

fn node_secret(i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0x51 + i as u8;
    s[1] = 0x2A;
    s
}

fn credentials_for(
    i: usize,
    env: &LiveEnv,
    envelope_hash: [u8; 32],
    checkpoint_every_rounds: u32,
) -> JoinCredentials {
    let roster: Vec<[u8; 32]> = (0..NUM_WORKERS)
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
        iroh: None,
        presign_base: Some(env.presign_base.clone()),
        engine: EngineParams {
            steps_per_round: STEPS_PER_ROUND,
            micro_batch: MICRO_BATCH,
            stall_rounds_max: 3,
            checkpoint_every_rounds,
            update_max_bytes: 1 << 20,
            corpus_seed: 7,
            corpus_shards: 4,
            corpus_tokens_per_shard: 256,
            corpus_seq_len: 8,
            corpus_vocab_clamp: GUEST_VOCAB,
            payload_retention_rounds: 16,
            corpus: None,
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

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
}

fn hex16(d: &[u8; 16]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Spawn `NUM_WORKERS` local workers, run the kill→park→rejoin churn with the given checkpoint
/// cadence, and return `(digests[peer][round], rejoined_rounds, drop_index)`.
async fn run_churn(
    env: &LiveEnv,
    tag: &str,
    checkpoint_every_rounds: u32,
) -> (Vec<BTreeMap<u64, [u8; 16]>>, BTreeMap<u64, [u8; 16]>, usize) {
    let worker_bin = ensure_built();
    let module_path = tiny_llama_wasm_path();
    let module_bytes = std::fs::read(&module_path).expect("read tiny_llama.wasm");
    let global_batch = NUM_WORKERS as u32 * STEPS_PER_ROUND * MICRO_BATCH;
    let drop_index = NUM_WORKERS - 1;
    let last_round = ROUNDS - 1;

    let run_id = format!("run-r-resync-{tag}-{}", now_secs());
    let envelope = author_envelope(&run_id, &module_path, &module_bytes, global_batch);
    envelope.validate().expect("envelope validates");
    let author = SigningKey::from_bytes(&[0x2Au8; 32]);
    let frozen = envelope.freeze(&author).expect("freeze envelope");
    frozen.verify().expect("verify frozen envelope");
    let envelope_hash = frozen.hash().0;

    let egress = EgressClient::new(EgressConfig::default()).expect("egress client");
    let request = create_run_request(&envelope, &frozen, &module_bytes, global_batch);
    let (status, text) = post_json(&egress, env, &format!("{}/runs", env.ws_base), &request).await;
    assert_eq!(status, 201, "POST /runs: {text}");
    let wire = to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope");
    println!("[{tag}] created {run_id} (checkpoint_every_rounds={checkpoint_every_rounds})");

    let build_sup = |i: usize| {
        let mut cfg = TrainClientConfig::new(&worker_bin);
        cfg.env.push((
            "DAEMON_TRAIN_MODULE".to_string(),
            module_path.to_string_lossy().into_owned(),
        ));
        let _ = i;
        cfg.spawn_timeout = Duration::from_secs(60);
        cfg.op_timeout = Duration::from_secs(180);
        Arc::new(TrainSupervisor::new(cfg))
    };

    let mut supervisors = Vec::new();
    let mut streams: Vec<Option<tokio::sync::mpsc::UnboundedReceiver<Event>>> = Vec::new();
    for i in 0..NUM_WORKERS {
        let sup = build_sup(i);
        let elig = sup
            .assess(wire.clone())
            .await
            .unwrap_or_else(|e| panic!("peer {i} assess: {e}"));
        assert!(elig.eligible, "peer {i} eligible: {:?}", elig.reasons);
        let creds = credentials_for(i, env, envelope_hash, checkpoint_every_rounds)
            .to_bytes()
            .expect("encode credentials");
        let rx = sup
            .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
            .await
            .unwrap_or_else(|e| panic!("peer {i} join_streaming: {e}"));
        supervisors.push(sup);
        streams.push(Some(rx));
    }

    let state_url = format!("{}/runs/{run_id}/state", env.ws_base);
    let mut digests: Vec<BTreeMap<u64, [u8; 16]>> = vec![BTreeMap::new(); NUM_WORKERS];
    let mut rejoined_rounds: BTreeMap<u64, [u8; 16]> = BTreeMap::new();
    let mut rejoined_stream: Option<tokio::sync::mpsc::UnboundedReceiver<Event>> = None;
    let mut dropped = false;
    let mut rejoined = false;

    let budget = Duration::from_secs(240 + ROUNDS * (u64::from(ROUND_TIMEOUT_S) + 30));
    let deadline = Instant::now() + budget;

    'collect: loop {
        assert!(
            Instant::now() < deadline,
            "[{tag}] run budget exceeded; rounds: {:?} rejoined: {:?}",
            digests.iter().map(BTreeMap::len).collect::<Vec<_>>(),
            rejoined_rounds.keys().collect::<Vec<_>>()
        );
        for (i, slot) in streams.iter_mut().enumerate() {
            if let Some(rx) = slot.as_mut() {
                while let Ok(ev) = rx.try_recv() {
                    match ev {
                        Event::RoundOutcome { round, digest, .. } => {
                            digests[i].insert(round, digest);
                        }
                        Event::Error { class, detail } => {
                            eprintln!("[{tag}] peer {i} ERROR {class:?}: {detail}")
                        }
                        Event::Warning { class, detail } => {
                            eprintln!("[{tag}] peer {i} WARN [{class}]: {detail}")
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
                        digests[drop_index].insert(round, digest);
                        rejoined_rounds.insert(round, digest);
                    }
                    Event::ResyncProgress {
                        round,
                        from_checkpoint,
                        replayed,
                        total,
                    } => {
                        eprintln!("[{tag}] rejoiner RESYNC round {round} from checkpoint {from_checkpoint} ({replayed}/{total})");
                    }
                    Event::Warning { class, detail } => {
                        eprintln!("[{tag}] rejoiner WARN [{class}]: {detail}")
                    }
                    _ => {}
                }
            }
        }

        // Kill the drop target once it reported DROP_AFTER_ROUND.
        if !dropped && digests[drop_index].contains_key(&DROP_AFTER_ROUND) {
            println!("[{tag}] CHURN: killing peer {drop_index} after round {DROP_AFTER_ROUND}");
            supervisors[drop_index]
                .leave(run_id.clone(), LeaveMode::Immediate)
                .await
                .ok();
            supervisors[drop_index].shutdown().await;
            streams[drop_index] = None;
            dropped = true;
        }

        // Rejoin once the coordinator parks (floor breach → waiting).
        if dropped && !rejoined {
            let (_st, state) = get_json(&egress, env, &state_url).await;
            if state["data"]["phase"] == serde_json::json!("waiting") {
                println!(
                    "[{tag}] CHURN: coordinator parked (round {}); rejoining peer {drop_index}",
                    state["data"]["round"]
                );
                let sup = &supervisors[drop_index];
                let elig = sup
                    .assess(wire.clone())
                    .await
                    .expect("re-assess on respawn");
                assert!(elig.eligible, "respawned peer eligible");
                let creds =
                    credentials_for(drop_index, env, envelope_hash, checkpoint_every_rounds)
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

        // Done when every peer (incl. the rejoiner) reported the last round.
        if dropped && rejoined && (0..NUM_WORKERS).all(|i| digests[i].contains_key(&last_round)) {
            break 'collect;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    assert!(
        dropped && rejoined,
        "[{tag}] kill→park→rejoin cycle exercised"
    );

    // Wait for Finished (best-effort).
    let finish_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_st, state) = get_json(&egress, env, &state_url).await;
        if state["data"]["finished"] == serde_json::Value::Bool(true) {
            println!("[{tag}] final DO state: {}", state["data"]);
            break;
        }
        if Instant::now() >= finish_deadline {
            break;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    for sup in &supervisors {
        sup.leave(run_id.clone(), LeaveMode::Immediate).await.ok();
        sup.shutdown().await;
    }
    (digests, rejoined_rounds, drop_index)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resync_rejoiner_is_byte_identical() {
    let Some(env) = live_env() else {
        eprintln!("SKIP checkpoint_resync: set SWARM_LIVE_WS_URL (e.g. http://127.0.0.1:8795/api/v1/swarm)");
        return;
    };
    let (digests, rejoined_rounds, drop_index) = run_churn(&env, "resync", 2).await;

    // Transcript.
    for round in 0..ROUNDS {
        let cols: Vec<String> = (0..NUM_WORKERS)
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

    // The headline: every peer that reported a round — INCLUDING the rejoiner's post-resync rounds —
    // agrees byte-for-byte. B4's exclusion is gone: checkpoint-resync makes the rejoin byte-identical.
    for round in 0..ROUNDS {
        let mut reference: Option<(usize, [u8; 16])> = None;
        for (i, d) in digests.iter().enumerate() {
            if let Some(dig) = d.get(&round) {
                match reference {
                    None => reference = Some((i, *dig)),
                    Some((ri, rd)) => assert_eq!(
                        rd, *dig,
                        "round {round}: peer {ri} and peer {i} digests diverge — resync BROKEN"
                    ),
                }
            }
        }
    }
    // The rejoiner actually re-contributed POST-drop rounds via resync (not a vacuous pass).
    assert!(
        !rejoined_rounds.is_empty(),
        "rejoiner contributed post-resync rounds"
    );
    assert!(
        rejoined_rounds.keys().any(|&r| r > DROP_AFTER_ROUND),
        "rejoiner's post-resync rounds are past the drop point"
    );
    assert!(
        digests[drop_index].contains_key(&(ROUNDS - 1)),
        "rejoiner reported the last round byte-identically after resync"
    );
    println!(
        "checkpoint-resync GREEN: rejoiner byte-identical across {} post-resync round(s): {:?}",
        rejoined_rounds.len(),
        rejoined_rounds.keys().collect::<Vec<_>>()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fresh_state_rejoin_still_finishes() {
    let Some(env) = live_env() else {
        eprintln!("SKIP checkpoint_resync fresh-state: set SWARM_LIVE_WS_URL");
        return;
    };
    // checkpoint_every_rounds = 0 ⇒ NO checkpoint is ever published, so the rejoiner finds no pointer
    // and falls back to fresh-state (the pre-P3 / §9 first-epoch behavior). The run must still finish.
    let (digests, _rejoined, drop_index) = run_churn(&env, "fresh", 0).await;
    // Survivors completed every round.
    for i in (0..NUM_WORKERS).filter(|&i| i != drop_index) {
        assert_eq!(
            digests[i].len() as u64,
            ROUNDS,
            "survivor {i} completed every round with no checkpoint published"
        );
    }
    println!("fresh-state fallback GREEN: no checkpoint published, churn survived, run finished");
}
