// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The genesis-native in-process whole-run** (the worker's supported self-driven join): the
// compute@2 trainer (`tiny_llama.wasm` — a real Burn LLaMA, det-lane ingest in-guest, kind-0
// byte staging) under a **genesis envelope v2**, coordinated by the run's REAL coordinator
// (`coordinator_quorum.wasm`, pinned in the genesis role set, resolved by content hash from the
// artifact map, and run in-process under the same major-2 driver), through the real worker
// protocol: probe → assess (claim funnel + the worker role's device_min pre-screen + role
// grants) → join → two full barrier rounds (train → guest-sealed commit → record → ingest) →
// the guest's tag-4 det digest. Consensus never runs outside the sandboxed, content-addressed
// module — the native-tick drive this file used to pin retired with the v1-envelope (device-min admission pre-screen)
// form, which now refuses typed at assess (worker_protocol.rs pins the refusal).
//
// The run also proves the inline replay soak (refactor §12.6): the worker re-drives its own
// recorded journal through the §8.7 engine before reporting the round outcome — a diverging
// run is a join FAILURE (the `replay_decisions` metric is the green receipt).
//
// In-process identity contract (session module docs): the worker resolves every signing key
// against the identity keystore this fixture hands it (DAEMON_VHC_IDENTITY_DIR) — CSPRNG per-run
// keys, never derived from the run id. The author therefore reads the coordinator run key's peer
// id OUT of that same store and names it as the genesis `SingleKey` coordinator identity — a
// mismatch refuses every coordinator frame at the authority judgment.

// Dev/test harness: shells cargo for the guest build (same pattern as worker_protocol.rs); the
// env/spawn bans target the shipped node, so they are allowed file-wide here.
#![allow(clippy::disallowed_methods)]
// The in-process self-driven join is harness machinery (`--features harness` builds the worker
// with it); a default worker build refuses JoinRun typed, so this suite only runs on the
// harness-featured lane.
#![cfg(feature = "harness")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;
use daemon_vhc_proto::genesis::{
    ChannelDecl, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, GenesisEnvelope, Hash, PeerId, Seed, SignedEnvelope,
    SigningKey, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};
use daemon_vhc_session::keystore::{VhcKeystore, IDENTITY_DIR_ENV};
use daemon_vhc_session::protocol::{Event, JoinPolicy, PolicyMode};
use daemon_vhc_supervisor::{TrainClientConfig, TrainSupervisor};

const RUN_ID: &str = "genesis-t2";

/// The identity keystore this test authors the genesis FROM and hands the worker (by the
/// directory reference the node would pass): the trainer/coordinator per-run keys are CSPRNG
/// material minted here, and the genesis pins their public halves.
struct IdentityFixture {
    dir: tempfile::TempDir,
    worker_peer: PeerId,
    coordinator_peer: PeerId,
}

fn identity_fixture() -> &'static IdentityFixture {
    static FIXTURE: std::sync::OnceLock<IdentityFixture> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        let dir = tempfile::tempdir().expect("identity tempdir");
        let store = VhcKeystore::open(dir.path()).expect("open keystore");
        let worker_peer = peer_id(
            &store
                .run_signing_key(RUN_ID, "trainer", 1)
                .expect("mint trainer run key"),
        );
        let coordinator_peer = peer_id(
            &store
                .run_signing_key(RUN_ID, "coordinator", 1)
                .expect("mint coordinator run key"),
        );
        IdentityFixture {
            dir,
            worker_peer,
            coordinator_peer,
        }
    })
}
/// The t2 drive geometry (must match the worker's in-process join constants).
const STEPS_PER_ROUND: u32 = 2;
const SEQ_LEN: u32 = 9;
const GLOBAL_BATCH: u32 = 2;

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
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

fn module_path(name: &str) -> PathBuf {
    BUILD.call_once(|| {
        let status = StdCommand::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"))
}

/// The trainer's canonical parameter element counts for the tiny parity-shape model below
/// (`ModelCfg::param_numels` — tok, per-block 9 params, final norm).
fn param_numels() -> Vec<usize> {
    let (d, qdim, hidden, vocab) = (64usize, 64usize, 128usize, 64usize);
    let mut out = vec![vocab * d];
    out.extend([
        d,
        d * qdim,
        d * qdim,
        d * qdim,
        qdim * d,
        d,
        d * hidden,
        d * hidden,
        hidden * d,
    ]);
    out.push(d);
    out
}

/// A deterministic small init (the guest asserts the flat length against its layout).
fn deterministic_init(total: usize) -> Vec<f32> {
    let mut s = 0x5EED_C0DEu64;
    (0..total)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            #[allow(clippy::cast_precision_loss)]
            let v = ((s >> 33) % 2001) as f32;
            (v - 1000.0) / 20000.0 // [-0.05, 0.05]
        })
        .collect()
}

/// The trainer role's opaque config: the compute@2 trainer's `GuestCfg` map (tiny parity-shape
/// model, sparse_loco profile, matched deterministic init). The `peer`/`roster` are the
/// worker's DERIVED peer — the coordinator's record entries carry the joined peer identity, and
/// the guest's payload map is keyed by it.
fn trainer_role_config() -> Value {
    let peer = identity_fixture().worker_peer;
    let model = Value::Map(vec![
        (Value::from("d_model"), Value::from(64u32)),
        (Value::from("n_layers"), Value::from(1u32)),
        (Value::from("n_heads"), Value::from(4u32)),
        (Value::from("head_dim"), Value::from(16u32)),
        (Value::from("vocab"), Value::from(64u32)),
        (Value::from("seq_len"), Value::from(SEQ_LEN)),
        (Value::from("ffn_mult"), Value::from(2u32)),
        (Value::from("rope_theta"), Value::from(10_000.0f64)),
        (Value::from("rmsnorm_eps"), Value::from(1e-5f64)),
        (Value::from("lr"), Value::from(4e-4f64)),
        (Value::from("beta1"), Value::from(0.9f64)),
        (Value::from("beta2"), Value::from(0.95f64)),
        (Value::from("adam_eps"), Value::from(1e-8f64)),
        (Value::from("wd"), Value::from(0.1f64)),
    ]);
    let profile = Value::Map(vec![
        (Value::from("h"), Value::from(3u32)),
        (Value::from("ef_decay"), Value::from(0.95f64)),
        (Value::from("chunk"), Value::from(64u32)),
        (Value::from("topk"), Value::from(8u32)),
        (Value::from("bits"), Value::from(2u32)),
        (Value::from("outer_alpha"), Value::from(1.0f64)),
        (Value::from("clip"), Value::Bool(false)),
    ]);
    let total: usize = param_numels().iter().sum();
    let init = deterministic_init(total);
    Value::Map(vec![
        (Value::from("model"), model),
        (Value::from("peer"), Value::Bytes(peer.0.to_vec())),
        (
            Value::from("roster"),
            Value::Array(vec![Value::Bytes(peer.0.to_vec())]),
        ),
        (Value::from("steps_per_round"), Value::from(STEPS_PER_ROUND)),
        (Value::from("micro_batch"), Value::from(1u32)),
        (Value::from("stall_rounds_max"), Value::from(2u32)),
        (Value::from("profile"), profile),
        (
            Value::from("init"),
            Value::serialized(&init).expect("init value"),
        ),
    ])
}

/// The coordinator role's opaque config: an authored `RunConfig` + genesis `CoordinatorState`,
/// exactly the coordinator-quorum guest's `da_init` shape (`{state: …}`; event-driven synthetic
/// clock — phase deadlines effectively infinite, the ready-heartbeat fast path opens rounds).
fn coordinator_role_config() -> Value {
    let run_config = RunConfig {
        run_id: RUN_ID.to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: daemon_vhc_proto::CapabilitySet::new(),
        min_peers: 1,
        max_peers: 4,
        warmup_s: 1_000_000,
        round_train_max_s: 1_000_000,
        round_witness_s: 1_000_000,
        cooldown_s: 1_000_000,
        epoch_rounds: 0,
        stall_rounds_max: 2,
        global_batch: daemon_vhc_proto::envelope::GlobalBatch {
            start: GLOBAL_BATCH,
            end: GLOBAL_BATCH,
            ramp_rounds: 1,
        },
        stop: daemon_vhc_proto::envelope::StopCondition::Rounds(1_000_000),
        steps_per_round: STEPS_PER_ROUND,
        seq_len: u64::from(SEQ_LEN),
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let state = CoordinatorState::new(run_config, Seed([0x33; 32]), 0);
    Value::Map(vec![(
        Value::Text("state".into()),
        Value::serialized(&state).expect("state to cbor value"),
    )])
}

/// Author + freeze the genesis envelope v2: coordinator + trainer roles pinning the REAL built
/// blobs by content hash (file:// URLs — the worker fetches BOTH through the artifact resolver),
/// the derived `SingleKey` coordinator identity, and the declared authority topology.
fn genesis_wire() -> Vec<u8> {
    let coord_wasm = std::fs::read(module_path("coordinator_quorum")).expect("coordinator blob");
    let worker_wasm = std::fs::read(module_path("tiny_llama")).expect("trainer blob");
    let coord_identity = identity_fixture().coordinator_peer;

    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "coordinator.wasm".to_string(),
        SnapshotArtifact {
            url: format!("file://{}", module_path("coordinator_quorum").display()),
            blake3: blake3_hash(&coord_wasm),
            size: None,
        },
    );
    artifacts.insert(
        "worker.wasm".to_string(),
        SnapshotArtifact {
            url: format!("file://{}", module_path("tiny_llama").display()),
            blake3: blake3_hash(&worker_wasm),
            size: None,
        },
    );

    // The roles' channel table: the control channel both modules' manifests name (the
    // manifest ⊆ admitted-channels check is §9.4 step 6 — omitting channel 0 is a typed
    // GrantsExceedLane refusal). Numeric quotas stay unset, inheriting the lane ceiling.
    let control_channel = |name: &str| RoleGrants {
        channels: vec![ChannelDecl {
            id: 0,
            name: name.into(),
            class: 0,     // authoritative
            direction: 2, // bidirectional
            max_frame_bytes: 1 << 20,
            rate_per_min: 600,
            spool_frames: Some(256),
            replay_window: Some(1024),
            per_sender_quota: Some(64),
        }],
        ..RoleGrants::default()
    };
    let mut roles = BTreeMap::new();
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
            lane: "coordinator".into(),
            module: "coordinator.wasm".into(),
            abi: "vhc@2".into(),
            config: coordinator_role_config(),
            grants: control_channel("control"),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );
    roles.insert(
        "trainer".to_string(),
        RoleEntry {
            lane: "trainer".into(),
            module: "worker.wasm".into(),
            abi: "vhc@2".into(),
            config: trainer_role_config(),
            grants: control_channel("control"),
            device_min: daemon_vhc_proto::DeviceMinimums {
                gpu: Some(1), // optional
                ram_bytes: Some(1 << 20),
                ..Default::default()
            },
        },
    );

    let env = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: RUN_ID.to_string(),
            min_peers: 1,
            max_peers: 4,
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        // The run's declared trust topology (D1's typed AuthorityConfig, encoded into the opaque
        // section the host never interprets): SingleKey over the DERIVED coordinator identity.
        authority: AuthorityConfig {
            topology: Topology::SingleKey(SingleKey::new(coord_identity)),
            records_channel: DEFAULT_RECORDS_CHANNEL,
        }
        .encode(),
        transport: TransportSelection::default(),
        identities: Identities {
            coordinator: Some(coord_identity),
            coordinator_set: Vec::new(),
            upgrade_authority: Vec::new(),
        },
    };
    let author = SigningKey::from_bytes(&[0x42; 32]);
    let frozen = env.freeze(&author).expect("freeze genesis");
    let wire = SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    to_canonical_vec(&wire).expect("wire")
}

/// Public run data must not derive any signing key: reconstruct the retired derivation shape
/// (a blake3 of the public run label) and assert it matches NEITHER identity the keystore minted
/// — the coordinator identity the genesis names, nor the worker peer the trainer config pins.
/// (The attach/authority refusal of such a key is pinned in the session attach suite; here the
/// worker's own fixture proves the identities are not reconstructible from the label.)
#[test]
fn run_label_derived_keys_match_no_minted_identity() {
    for prefix in ["vhc-worker", "vhc-coordinator"] {
        let derived_seed = *blake3::hash(format!("{prefix}/{RUN_ID}").as_bytes()).as_bytes();
        let derived = peer_id(&SigningKey::from_bytes(&derived_seed));
        assert_ne!(derived, identity_fixture().worker_peer);
        assert_ne!(derived, identity_fixture().coordinator_peer);
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

/// probe → assess → join → two coordinator-driven barrier rounds → replay-soaked digest.
#[tokio::test]
async fn worker_joins_and_runs_rounds_under_the_coordinator() {
    let wire = genesis_wire(); // also builds the guests
    let mut cfg = TrainClientConfig::new(env!("CARGO_BIN_EXE_daemon-vhc-worker").to_string());
    cfg.env = vec![
        // The owner's node-side lane choice (§9.6 numbers-are-config): CPU-admitting t2 lane.
        // NO module-source override: both blobs resolve by content hash from the artifact map.
        ("DAEMON_VHC_LANE_GPU_OPTIONAL".to_string(), "1".to_string()),
        // The identity-store reference the node would pass: the worker resolves its per-run and
        // coordinator keys against the SAME store this test authored the genesis from.
        (
            IDENTITY_DIR_ENV.to_string(),
            identity_fixture().dir.path().display().to_string(),
        ),
    ];
    cfg.spawn_timeout = Duration::from_secs(30);
    cfg.op_timeout = Duration::from_secs(300);
    let sup = TrainSupervisor::new(cfg);

    // Assess through the REAL funnel: the genesis worker role's device_min feeds stage 3, its
    // grant list stage 4.0; the claim is the trainer's SDK-derived declaration.
    let elig = sup.assess(wire, None).await.expect("assess");
    assert!(
        elig.eligible,
        "the compute@2 trainer admits under the t2 lane: {:?}",
        elig.reasons
    );

    // Join: the worker's v2 session — pump attach + the run's REAL coordinator in-process.
    let mut events = sup
        .join_streaming(RUN_ID, "local://coordinator", vec![], policy(), None)
        .await
        .expect("join");

    // The event stream: RunPhase(train) → Metric(replay_decisions) → RoundOutcome(final digest).
    let mut replay_decisions = None;
    let mut outcome = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    while outcome.is_none() {
        let ev = tokio::time::timeout_at(deadline, events.recv())
            .await
            .expect("worker event stream stalled")
            .expect("worker event stream closed early");
        match ev {
            Event::Metric { name, value } if name == "replay_decisions" => {
                replay_decisions = Some(value);
            }
            Event::RoundOutcome {
                round,
                committed,
                ingested,
                stalled,
                digest,
                ..
            } => {
                assert_eq!(round, 1, "two rounds ran (0 and 1)");
                assert_eq!((committed, ingested, stalled), (1, 1, false));
                assert_ne!(
                    digest, [0u8; 16],
                    "a real det-lane digest (the guest's tag-4)"
                );
                outcome = Some(digest);
            }
            Event::Error { detail, .. } => panic!("worker error: {detail}"),
            _ => {}
        }
    }

    // The inline §12.6 replay soak ran and reproduced every decision
    // (2 rounds × 3 publishes: theta + commitment + digest).
    assert_eq!(
        replay_decisions,
        Some(6.0),
        "the recorded journal re-drove bit-for-bit before the outcome was reported"
    );
}
