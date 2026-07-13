// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **A3 live worker-subprocess e2e — the Merge-2 rehearsal.**
//!
//! Extends the Merge-1 `ws_live_do.rs` + `seed_run.mjs` scaffold into the full node↔cloud↔worker
//! loop: a wrangler-dev `RunCoordinatorDO` + registry, N REAL `daemon-train-worker` subprocesses
//! (built with the `swarm-net` feature) spawned via `TrainSupervisor`, each running a `RoundEngine`
//! over `DualPlane(WsControlPlane[, IrohGossip])` + the **object-proxy R2 store** (payload PUT/GET
//! through the live presign endpoint), the tiny-llama guest as the experiment, N≥5 rounds:
//!
//! - per-round det digests **byte-identical** across workers;
//! - the worker→node **event pump** visible in `SwarmService` state (`swarm.db` round progression);
//! - one worker dropped mid-run: the coordinator drops it after K record-absences, the survivors
//!   complete every round in agreement (run-level recovery), and the supervisor respawns the child
//!   on the next command (node-side supervision, §10.3/§13);
//! - the **declared RunConfig** (Merge-1 Decision 1) drives the DO's phase timings live.
//!
//! Env-gated (needs wrangler-dev; skips cleanly in the offline gate). **Every endpoint is
//! configurable** so the same harness targets wrangler-dev (the Merge-2 gate) or the real
//! `daemon-swarm-dev` workers.dev deployment + M1-mini relay without a code change:
//!
//! ```text
//! SWARM_LIVE_WS_URL       coordinator/registry base   (e.g. http://127.0.0.1:8795/api/v1/swarm)
//! SWARM_LIVE_PRESIGN_BASE presign base                (default: SWARM_LIVE_WS_URL)
//! SWARM_LIVE_RELAY_URL    optional iroh relay URL — set ⇒ dual-plane (WS + iroh) workers
//! SWARM_LIVE_ORG/ACTOR    internal identity headers   (default org_live / key:live)
//! ```
//!
//! Drive it (after `pnpm -C apps/swarm dev` on the daemon-cloud branch):
//! ```text
//! SWARM_LIVE_WS_URL=http://127.0.0.1:8795/api/v1/swarm \
//!   cargo test -p daemon-swarm-e2e --test ws_live_workers -- --nocapture --test-threads 1
//! ```

// Test harness: builds the worker binary + guests via cargo (the sanctioned dev-tool exception,
// mirroring `live_transport.rs`), reads operator-local files, and prints progress.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_swarm_proto::{peer_id, to_canonical_vec, SigningKey};
use daemon_swarm_run::protocol::{
    EngineParams, Event, IrohCredentials, IrohRosterPeer, JoinCredentials, JoinPolicy, LeaveMode,
    PolicyMode, WsAuthSpec,
};
use daemon_train_client::{TrainClientConfig, TrainSupervisor};
use daemon_train_sdk::models::TinyLlamaCfg;

// ---- knobs ---------------------------------------------------------------------------------------

/// 4 workers: worker 0 joins through the `SwarmService` pump (payload-free wire events → swarm.db),
/// workers 1..3 stream digests directly, worker 3 is the mid-run kill target — so byte-identical
/// digest comparison covers TWO full-run streams (1, 2) on every round plus the kill target's
/// pre-drop rounds.
const NUM_WORKERS: usize = 4;
const NUM_ROUNDS: u64 = 8;
/// Kill worker `DROP_INDEX` once it has reported this round (the drop drill).
const DROP_INDEX: usize = 3;
const DROP_AFTER_ROUND: u64 = 1;
const GUEST_VOCAB: u32 = 64;
/// Declared RunConfig (Merge-1 Decision 1) — replaces the DO's T0 constants live.
const DECLARED_WARMUP_S: u64 = 8;
/// Tight on purpose: after the mid-run kill, every round until the K-absence drop waits out the
/// full round timeout for the dead peer's commitment (plus the T0 30 s witness window) — the
/// declared timeout (Decision 1) is what keeps the drill's wall time bounded.
const DECLARED_ROUND_TIMEOUT_S: u64 = 20;
const DECLARED_COOLDOWN_S: u64 = 1;
/// Must divide evenly across the roster's per-round intervals: `NUM_WORKERS × steps_per_round ×
/// micro_batch` (4 × 2 × 2) — a peer whose seed-shuffled interval underfills its steps errors out
/// of the round loop instead of committing.
const DECLARED_GLOBAL_BATCH: u32 = 16;
/// Overall wall budget for the run (WAN-ish paths through wrangler-dev are slow but bounded:
/// ~3 timeout-paced rounds × ~51 s while the dead peer ages out, the rest event-driven).
const RUN_BUDGET: Duration = Duration::from_secs(600);

// ---- env-configurable endpoints (the Merge-2 → real-deployment swap is config-only) --------------

struct LiveEnv {
    ws_base: String,
    presign_base: String,
    relay_url: Option<String>,
    org: String,
    actor: String,
}

fn live_env() -> Option<LiveEnv> {
    let ws_base = std::env::var("SWARM_LIVE_WS_URL").ok()?;
    Some(LiveEnv {
        presign_base: std::env::var("SWARM_LIVE_PRESIGN_BASE").unwrap_or_else(|_| ws_base.clone()),
        relay_url: std::env::var("SWARM_LIVE_RELAY_URL").ok(),
        org: std::env::var("SWARM_LIVE_ORG").unwrap_or_else(|_| "org_live".into()),
        actor: std::env::var("SWARM_LIVE_ACTOR").unwrap_or_else(|_| "key:live".into()),
        ws_base,
    })
}

// ---- guest + worker binary build (mirrors live_transport.rs / worker_protocol.rs) ----------------

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

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.parent().unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

/// Build the guest wasm (warn-and-rebuild guard semantics — the freshly built module is used) and
/// the `daemon-train-worker` binary WITH the `swarm-net` feature (the live attach).
fn ensure_built() -> PathBuf {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_err() {
            let status = Command::new("cargo")
                .current_dir(guests_root())
                .env_remove("CARGO_TARGET_DIR")
                .env("RUSTFLAGS", guest_remap_rustflags())
                .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
                .status()
                .expect("run cargo for guests (dev shell provides the wasm target)");
            assert!(status.success(), "building guest modules failed");
        }
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

// ---- run authoring (the same §6.1 chain as swarm-local) ------------------------------------------

fn tiny_cfg() -> ciborium::value::Value {
    ciborium::value::Value::serialized(&TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9, // corpus seq_len (8) + 1
        vocab: GUEST_VOCAB,
        profile: "sparse_loco".to_string(),
        ..TinyLlamaCfg::default()
    })
    .expect("tiny-llama config serializes")
}

fn author_envelope(run_id: &str, module_path: &Path, module_bytes: &[u8]) -> Envelope {
    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        "tiny_llama.wasm".to_string(),
        Artifact {
            url: format!("file://{}", module_path.display()),
            blake3: daemon_swarm_proto::blake3_hash(module_bytes),
        },
    );
    Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: run_id.to_string(),
            // min_peers == NUM_WORKERS so ALL workers are admitted during `WaitingForMembers`
            // (§6.2: a join arriving after the warmup transition is staged `pending` until the next
            // epoch — with `epoch_rounds = 0` it would never materialize mid-run; an early e2e
            // iteration hit exactly that). The K-absence drop then breaches the floor and the tick
            // parks the run back in WaitingForMembers — which IS the recovery drill: the node
            // supervisor respawns the killed worker, it RE-JOINS (a previously-Dropped member may
            // rejoin, §6.5), warmup re-runs, and the run completes.
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
            manifest: "tiny_llama.wasm".to_string(), // manifest name must resolve; reuse the artifact
            steps_per_round: 2,
            global_batch: GlobalBatch {
                start: DECLARED_GLOBAL_BATCH,
                end: DECLARED_GLOBAL_BATCH,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(NUM_ROUNDS),
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
            warmup: DECLARED_WARMUP_S as u32,
            round_train_max: DECLARED_ROUND_TIMEOUT_S as u32,
            round_witness: 30,
            cooldown: DECLARED_COOLDOWN_S as u32,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 3,
            payload_retention_rounds: 16,
        },
    }
}

// ---- registry HTTP (EgressClient; internal dev identity) -----------------------------------------

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
    let value = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    (status, value)
}

/// The `CreateRunRequest` with the DECLARED RunConfig (Decision 1) — the authoring half emits it,
/// the registry forwards it verbatim to the DO `init`.
fn create_run_request(
    envelope: &Envelope,
    frozen: &daemon_swarm_proto::FrozenEnvelope,
    module_bytes: &[u8],
) -> serde_json::Value {
    use base64::Engine as _;
    serde_json::json!({
        "run_id": envelope.run.run_id,
        "schema": ENVELOPE_SCHEMA_MAJOR,
        "proto_version": daemon_swarm_proto::SWARM_PROTO_VERSION,
        "envelope_b64": base64::engine::general_purpose::STANDARD.encode(frozen.bytes()),
        "author_pubkey": frozen.signer().to_hex(),
        "signature": frozen.signature().to_hex(),
        "artifacts": [{
            "path": "tiny_llama.wasm",
            "blake3": daemon_swarm_proto::blake3_hash(module_bytes).to_hex(),
            "size": module_bytes.len(),
        }],
        "update_max_bytes": u64::from(envelope.requirements.update_mb_max) * 1024 * 1024,
        "min_peers": envelope.run.min_peers,
        "max_peers": envelope.run.max_peers,
        "rounds": NUM_ROUNDS,
        "warmup_timeout_s": DECLARED_WARMUP_S,
        "round_timeout_s": DECLARED_ROUND_TIMEOUT_S,
        "cooldown_s": DECLARED_COOLDOWN_S,
        "global_batch": DECLARED_GLOBAL_BATCH,
        "witness_target": 0,
    })
}

// ---- credentials authoring ------------------------------------------------------------------------

fn node_secret(i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0x51 + i as u8;
    s[1] = 0xA3;
    s
}

fn iroh_secret(i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0xB0 + i as u8;
    s[1] = 0xA3;
    s
}

fn credentials_for(i: usize, env: &LiveEnv, envelope_hash: [u8; 32]) -> JoinCredentials {
    let roster: Vec<[u8; 32]> = (0..NUM_WORKERS)
        .map(|j| peer_id(&SigningKey::from_bytes(&node_secret(j))).0)
        .collect();
    // Dual-plane iroh half only when a relay is configured: the workers are separate processes with
    // unknown bind ports, so roster reachability is relay-only (endpoint id + relay URL). The iroh
    // endpoint id is the ed25519 public key of the iroh secret (iroh SecretKey == ed25519 seed).
    let iroh = env.relay_url.as_ref().map(|relay| IrohCredentials {
        secret_key: iroh_secret(i),
        relay_urls: vec![relay.clone()],
        roster: (0..NUM_WORKERS)
            .map(|j| IrohRosterPeer {
                endpoint_id: peer_id(&SigningKey::from_bytes(&iroh_secret(j))).0,
                direct_addrs: Vec::new(),
                relay_url: Some(relay.clone()),
            })
            .collect(),
    });
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
            steps_per_round: 2,
            micro_batch: 2,
            stall_rounds_max: 3,
            checkpoint_every_rounds: 0,
            update_max_bytes: 1 << 20,
            corpus_seed: 7,
            corpus_shards: 4,
            corpus_tokens_per_shard: 256,
            corpus_seq_len: 8,
            corpus_vocab_clamp: GUEST_VOCAB,
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

// ---- the e2e --------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_workers_full_loop_via_wrangler_dev() {
    let Some(env) = live_env() else {
        eprintln!(
            "SKIP ws_live_workers: set SWARM_LIVE_WS_URL (e.g. http://127.0.0.1:8795/api/v1/swarm)"
        );
        return;
    };
    let worker_bin = ensure_built();
    let module_path = tiny_llama_wasm_path();
    let module_bytes = std::fs::read(&module_path).expect("read tiny_llama.wasm");

    // Unique run id per invocation (the registry rejects duplicates).
    let run_id = format!(
        "run-a3-e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs()
    );

    // -- author + freeze + create the run (declared RunConfig rides the create request) ----------
    let envelope = author_envelope(&run_id, &module_path, &module_bytes);
    envelope.validate().expect("envelope validates");
    let author = SigningKey::from_bytes(&[0xA1u8; 32]);
    let frozen = envelope.freeze(&author).expect("freeze envelope");
    frozen.verify().expect("verify frozen envelope");
    let envelope_hash = frozen.hash().0;
    let wire = to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope");

    let egress = EgressClient::new(EgressConfig::default()).expect("egress client");
    let request = create_run_request(&envelope, &frozen, &module_bytes);
    let (status, text) = post_json(&egress, &env, &format!("{}/runs", env.ws_base), &request).await;
    assert_eq!(status, 201, "POST /runs: {text}");
    println!("created {run_id} (declared warmup={DECLARED_WARMUP_S}s round={DECLARED_ROUND_TIMEOUT_S}s cooldown={DECLARED_COOLDOWN_S}s gb={DECLARED_GLOBAL_BATCH})");

    // -- spawn N REAL worker subprocesses via TrainSupervisor -------------------------------------
    let mut supervisors = Vec::new();
    for i in 0..NUM_WORKERS {
        let mut cfg = TrainClientConfig::new(&worker_bin);
        cfg.env.push((
            "DAEMON_TRAIN_MODULE".to_string(),
            module_path.to_string_lossy().into_owned(),
        ));
        cfg.op_timeout = Duration::from_secs(60);
        let sup = Arc::new(TrainSupervisor::new(cfg));
        let elig = sup.assess(wire.clone()).await.expect("assess run");
        assert!(elig.eligible, "worker {i} eligible: {:?}", elig.reasons);
        supervisors.push(sup);
    }

    // -- worker 0 joins THROUGH a SwarmService (the event-pump assertion path) --------------------
    let store = daemon_swarm_node::SwarmStore::open_in_memory().expect("in-memory swarm.db");
    let svc = Arc::new(daemon_swarm_node::SwarmService::new(
        daemon_swarm_node::SwarmServiceParts {
            config: daemon_swarm_run::config::SwarmConfig {
                enabled: true,
                ..Default::default()
            },
            store,
            worker: supervisors[0].clone(),
            feed: None,
            discovery: None,
        },
    ));
    svc.bind_self();
    // Persist the intent (so state reads have the run row), then join + pump.
    svc.store()
        .put_join_intent(
            &run_id,
            &env.ws_base,
            &daemon_api::SwarmPolicy {
                mode: daemon_api::SwarmPolicyMode::Always,
                vram_cap_mb: 0,
                duty_cycle_pct: 100,
                schedule: None,
            },
            None,
            &daemon_api::SwarmEligibility {
                eligible: true,
                reasons: Vec::new(),
                headroom: BTreeMap::new(),
            },
        )
        .expect("persist intent");
    let creds0 = credentials_for(0, &env, envelope_hash)
        .to_bytes()
        .expect("encode credentials");
    svc.join_and_pump(run_id.clone(), env.ws_base.clone(), creds0, policy())
        .await
        .expect("worker 0 join_and_pump");

    // -- workers 1..N join via join_streaming (digest collection off the raw event stream) --------
    let mut streams = Vec::new();
    for (i, sup) in supervisors.iter().enumerate().skip(1) {
        let creds = credentials_for(i, &env, envelope_hash)
            .to_bytes()
            .expect("encode credentials");
        let rx = sup
            .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
            .await
            .unwrap_or_else(|e| panic!("worker {i} join_streaming: {e}"));
        streams.push(rx);
    }

    // -- collect per-round digests until the survivors finish -------------------------------------
    // digests[worker][round] = 16-byte post-ingest digest from RoundOutcome. The kill target's
    // post-REJOIN digests land in `rejoined_digests` (a fresh-state rejoin does not fold the missed
    // history, so its digests are intentionally outside the byte-identity assertion; checkpoint
    // resync in the worker is a recorded follow-on).
    let mut digests: Vec<BTreeMap<u64, [u8; 16]>> = vec![BTreeMap::new(); NUM_WORKERS];
    let mut rejoined_digests: BTreeMap<u64, [u8; 16]> = BTreeMap::new();
    let mut rejoined_stream: Option<tokio::sync::mpsc::UnboundedReceiver<Event>> = None;
    let deadline = Instant::now() + RUN_BUDGET;
    let last_round = NUM_ROUNDS - 1;
    let state_url = format!("{}/runs/{run_id}/state", env.ws_base);
    let mut dropped = false;
    let mut rejoined = false;

    // Worker 0's digests come from swarm.db events (the pump path); 1..N from their streams.
    'collect: loop {
        assert!(
            Instant::now() < deadline,
            "run budget exceeded; digests so far: {:?} (rejoined: {:?})",
            digests.iter().map(BTreeMap::len).collect::<Vec<_>>(),
            rejoined_digests.len()
        );

        // Drain the direct streams (non-blocking).
        for (k, rx) in streams.iter_mut().enumerate() {
            let i = k + 1;
            while let Ok(ev) = rx.try_recv() {
                if let Event::RoundOutcome { round, digest, .. } = ev {
                    digests[i].insert(round, digest);
                }
            }
        }
        if let Some(rx) = rejoined_stream.as_mut() {
            while let Ok(ev) = rx.try_recv() {
                if let Event::RoundOutcome { round, digest, .. } = ev {
                    rejoined_digests.insert(round, digest);
                }
            }
        }

        // Read worker 0's progression from the durable log (proves the pump → swarm.db path).
        for ev in svc
            .store()
            .recent_events(&run_id, 512)
            .expect("recent events")
        {
            if let daemon_api::SwarmEvent::RoundOutcome { round, .. } = &ev {
                // The wire event carries no digest (payload-free, §10.4); mark presence and take
                // the digest from the store-independent stream comparison below only for 1..N.
                digests[0].entry(*round).or_insert([0u8; 16]);
            }
        }

        // Drop drill part 1: kill worker DROP_INDEX once it reported DROP_AFTER_ROUND.
        if !dropped && digests[DROP_INDEX].contains_key(&DROP_AFTER_ROUND) {
            println!(
                "dropping worker {DROP_INDEX} after round {DROP_AFTER_ROUND} (mid-run kill drill)"
            );
            supervisors[DROP_INDEX]
                .leave(run_id.clone(), LeaveMode::Immediate)
                .await
                .ok();
            supervisors[DROP_INDEX].shutdown().await;
            dropped = true;
        }

        // Drop drill part 2: once the coordinator has dropped the dead peer (K record-absences →
        // floor breach → the run parks in WaitingForMembers), the node-side supervision recovers
        // it: the supervisor respawns the child (lazy spawn), the run is re-assessed, and the peer
        // RE-JOINS (previously-Dropped members may rejoin, §6.5). The run then resumes.
        if dropped && !rejoined {
            let (status, state) = get_json(&egress, &env, &state_url).await;
            assert_eq!(status, 200, "GET /state during drop drill");
            if state["data"]["phase"] == serde_json::json!("waiting") {
                println!(
                    "coordinator dropped the killed peer (floor breach at round {}); rejoining",
                    state["data"]["round"]
                );
                let sup = &supervisors[DROP_INDEX];
                let elig = sup
                    .assess(wire.clone())
                    .await
                    .expect("re-assess on respawn");
                assert!(elig.eligible, "respawned worker eligible");
                let creds = credentials_for(DROP_INDEX, &env, envelope_hash)
                    .to_bytes()
                    .expect("encode credentials");
                let rx = sup
                    .join_streaming(run_id.clone(), env.ws_base.clone(), creds, policy())
                    .await
                    .expect("respawned worker rejoins");
                rejoined_stream = Some(rx);
                rejoined = true;
            }
        }

        // Finished when every SURVIVOR (direct streams 1..DROP_INDEX + worker 0 via the store)
        // reports the last round.
        let survivor_done = (1..DROP_INDEX).all(|i| digests[i].contains_key(&last_round))
            && digests[0].contains_key(&last_round);
        if survivor_done && dropped {
            break 'collect;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(rejoined, "the drop→park→rejoin cycle was exercised");

    // -- assertions --------------------------------------------------------------------------------
    // (1) N≥5 rounds completed by the survivors.
    assert!(
        digests[1].len() as u64 >= NUM_ROUNDS,
        "worker 1 ingested all {NUM_ROUNDS} rounds: got {:?}",
        digests[1].keys().collect::<Vec<_>>()
    );

    // (2) Per-round det digests byte-identical across every direct-stream worker that reported the
    // round (workers 1..DROP_INDEX for all rounds; the kill target for its pre-drop rounds).
    // Worker 0's wire events are payload-free by design (§10.4), so its agreement is transitive
    // through the coordinator's digest-mismatch detection (a mismatch would surface as a desync).
    for (round, d1) in &digests[1] {
        for i in 2..NUM_WORKERS {
            if let Some(d2) = digests[i].get(round) {
                assert_eq!(
                    d1, d2,
                    "round {round}: workers 1 and {i} disagree on the post-ingest digest"
                );
            }
        }
    }
    for i in 2..DROP_INDEX {
        assert_eq!(
            digests[i].len() as u64,
            NUM_ROUNDS,
            "survivor {i} reported every round"
        );
    }
    assert!(
        digests[DROP_INDEX].len() >= 2,
        "the dropped worker contributed at least rounds 0..=1 before the kill"
    );

    // (3) Event pump visible in SwarmService state: swarm.db carries the live round progression.
    let run_row = svc
        .store()
        .get_run(&run_id)
        .expect("get run")
        .expect("run row");
    assert!(
        run_row.last_round >= last_round,
        "swarm.db last_round reflects live progression (got {}, want >= {last_round})",
        run_row.last_round
    );
    let events = svc
        .store()
        .recent_events(&run_id, 512)
        .expect("recent events");
    let outcomes = events
        .iter()
        .filter(|e| matches!(e, daemon_api::SwarmEvent::RoundOutcome { .. }))
        .count();
    assert!(
        outcomes as u64 >= NUM_ROUNDS,
        "swarm.db carries >= {NUM_ROUNDS} RoundOutcome events (got {outcomes})"
    );
    let micro_batch_seen = events.iter().any(
        |e| matches!(e, daemon_api::SwarmEvent::Warning { class, .. } if class == "micro_batch"),
    );
    assert!(
        micro_batch_seen,
        "the additive MicroBatch telemetry reached swarm.db through the pump"
    );

    // (4) The run FINISHES despite the mid-run kill: the coordinator aged the dead peer out via K
    // record-absences (§6.4), parked on the floor breach, re-admitted the respawned peer (the
    // rejoin observed in the collect loop), and the stop condition (`Rounds(N)`) completed. The
    // cooldown→finished transition is alarm-driven, so poll /state briefly.
    let finish_deadline = Instant::now() + Duration::from_secs(60);
    let final_state = loop {
        let (status, state) = get_json(&egress, &env, &state_url).await;
        assert_eq!(status, 200, "GET /state");
        if state["data"]["finished"] == serde_json::Value::Bool(true) {
            break state;
        }
        assert!(
            Instant::now() < finish_deadline,
            "run did not reach Finished within 60s of the survivors' last round: {state}"
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    };
    let data = &final_state["data"];
    println!("final DO state: {data}");
    assert_eq!(
        data["round"],
        serde_json::json!(NUM_ROUNDS),
        "all rounds ran"
    );
    assert!(
        !digests[DROP_INDEX].contains_key(&last_round),
        "the killed worker's ORIGINAL stream must not carry the final round (it died mid-run)"
    );
    // (5) Node-side supervision already recovered the killed worker: the successful re-assess +
    // rejoin in the collect loop ran on a supervisor whose child had been shut down — the lazy
    // respawn path (§10.3/§13). Its post-rejoin engine contributed commitments (the run could not
    // have finished otherwise: min_peers == NUM_WORKERS, so rounds only advance with it back).
    println!(
        "rejoined worker reported {} post-rejoin round(s): {:?}",
        rejoined_digests.len(),
        rejoined_digests.keys().collect::<Vec<_>>()
    );

    // Cleanup.
    for sup in &supervisors {
        sup.leave(run_id.clone(), LeaveMode::Immediate).await.ok();
        sup.shutdown().await;
    }
    println!(
        "live worker e2e green: {NUM_ROUNDS} rounds, {} survivors agreeing, drop-recovery verified",
        NUM_WORKERS - 1
    );
}
