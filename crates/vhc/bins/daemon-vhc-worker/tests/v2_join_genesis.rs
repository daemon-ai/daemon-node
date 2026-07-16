// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The D3 cell-6 RETIREMENT pin** (decisions D3: the transitional native-coordinator adapter
// "exists only through D1 and is retired at D2"). Through D1 this file was the cell-6 positive
// whole-run; at D2 the adapter (`RunConfig::from_genesis`) was DELETED and the cell's verdict
// flipped: the supported v2×v2 combination is **cell 8** (wasm coordinator × envelope v2), pinned
// positive in the testkit's cell-8 whole-run lanes. What this test now pins, over the real worker
// protocol:
//
// - the genesis ASSESS path survives the retirement untouched (D1's lasting work): claim funnel +
//   the worker role's `device_min` pre-screen + the role grants threaded into the admission seam;
// - a genesis JOIN attempting the in-process native-coordinator drive is a TYPED refusal that
//   names the retirement and routes the author to the wasm coordinator (cell 8) — never a silent
//   fallback onto a coordinator shape that no longer exists.
//
// The v1-envelope twin (v2_join.rs, cell 5) stays green beside this until the Phase-E sunset.

// Dev/test harness: shells cargo for the guest build (same pattern as v2_join.rs); the env/spawn
// bans target the shipped node, so they are allowed file-wide here.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;
use daemon_vhc_proto::envelope::{Access, DeviceMinimums};
use daemon_vhc_proto::genesis::{
    ChannelDecl, Identities, RoleEntry, RoleGrants, RunSectionV2, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{to_canonical_vec, GenesisEnvelope, Hash, SignedEnvelope, SigningKey};
use daemon_vhc_sdk::models::TinyLlamaCfg;
use daemon_vhc_session::protocol::{Event, JoinPolicy, PolicyMode};
use daemon_vhc_supervisor::{TrainClientConfig, TrainSupervisor};

/// The run's human/registry label (`RunSectionV2::run_label`); the cryptographic RunId is the
/// genesis hash the worker derives itself.
const RUN_LABEL: &str = "cell6-genesis";

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

fn module_path() -> PathBuf {
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
    guests_root().join("target/wasm32-unknown-unknown/release/tiny_llama_v2.wasm")
}

/// The worker role's opaque config (`GuestCfg`): identical model/round shape to the cell-5 twin.
fn worker_role_config() -> Value {
    let model = Value::serialized(&TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    })
    .expect("model value");
    Value::Map(vec![
        (Value::from("model"), model),
        (Value::from("peer"), Value::Bytes(vec![7u8; 32])),
        (
            Value::from("roster"),
            Value::Array(vec![Value::Bytes(vec![7u8; 32])]),
        ),
        (Value::from("steps_per_round"), Value::from(2u32)),
        (Value::from("micro_batch"), Value::from(1u32)),
        (Value::from("stall_rounds_max"), Value::from(2u32)),
    ])
}

/// The coordinator role's opaque config — module-defined CBOR the HOST NEVER reads (the seam
/// rule, architecture §5.1). Since the cell-6 adapter's retirement at D2 no host-side decoder for
/// this exists at all: the wasm coordinator module is its only consumer. Authored here as an
/// arbitrary opaque map to prove the host treats it as such.
fn coordinator_role_config() -> Value {
    Value::Map(vec![
        (Value::from("steps_per_round"), Value::from(2u32)),
        (Value::from("warmup"), Value::from(1u32)),
        (Value::from("opaque-to-the-host"), Value::from(true)),
    ])
}

/// Author + freeze the genesis envelope, returning the `SignedEnvelope` wire bytes the worker's
/// `AssessRun` consumes (the schema sniff routes them down the genesis path).
fn genesis_wire() -> Vec<u8> {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "worker-mod".to_string(),
        SnapshotArtifact {
            // DAEMON_TRAIN_MODULE overrides the module source in the test drive (the explicit
            // dev/node-controlled override inside the signed path), so the pin is nominal.
            url: "file:///dev/null".into(),
            blake3: Hash([1; 32]),
            size: None,
        },
    );
    artifacts.insert(
        "coord-mod".to_string(),
        SnapshotArtifact {
            url: "file:///dev/null".into(),
            blake3: Hash([2; 32]),
            size: None,
        },
    );

    let mut roles = BTreeMap::new();
    roles.insert(
        "trainer".to_string(),
        RoleEntry {
            lane: "trainer".into(),
            module: "worker-mod".into(),
            abi: "vhc@2".into(),
            config: worker_role_config(),
            // The worker role's channel table: the control channel the module's manifest names
            // (the manifest ⊆ admitted-channels check is §9.4 step 6 — an envelope that omitted
            // channel 0 would be a typed GrantsExceedLane refusal). Every numeric quota is left
            // unset, inheriting the lane ceiling (tighten-only).
            grants: RoleGrants {
                channels: vec![ChannelDecl {
                    id: 0,
                    name: "control".into(),
                    class: 0,     // authoritative
                    direction: 2, // bidirectional
                    max_frame_bytes: 1 << 20,
                    rate_per_min: 600,
                    spool_frames: Some(256),
                    replay_window: Some(1024),
                    per_sender_quota: Some(64),
                }],
                ..RoleGrants::default()
            },
            device_min: DeviceMinimums {
                gpu: Some(1), // optional
                ram_bytes: Some(1 << 20),
                ..Default::default()
            },
        },
    );
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
            lane: "coordinator".into(),
            module: "coord-mod".into(),
            abi: "vhc@2".into(),
            config: coordinator_role_config(),
            grants: RoleGrants::default(),
            device_min: DeviceMinimums::default(),
        },
    );

    let env = GenesisEnvelope {
        run: RunSectionV2 {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: RUN_LABEL.into(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        roles,
        artifacts,
        // The opaque Authority section (D1 vocabulary): SingleKey under a nominal coordinator
        // identity — the host never interprets it; the SDK's AuthorityConfig does.
        authority: Value::Map(vec![
            (Value::from("topology"), Value::from("single-key")),
            (Value::from("coordinator"), Value::Bytes(vec![9u8; 32])),
        ]),
        transport: TransportSelection::default(),
        identities: Identities::default(),
    };
    let key = SigningKey::from_bytes(&[0x42; 32]);
    let frozen = env.freeze(&key).expect("freeze genesis");
    let wire = SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    to_canonical_vec(&wire).expect("wire")
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

/// **The post-D2 cell-6 verdict, pinned** (decisions D3: the transitional native-coordinator
/// adapter "exists only through D1 and is retired at D2"): the genesis ASSESS path — D1's
/// lasting work (device_min pre-screen + role grants threaded into the admission seam) — stays
/// green, but a JOIN attempting the retired in-process native-coordinator drive is a typed
/// refusal routing the author to the wasm coordinator (mixed-fleet cell 8, pinned positive in
/// the testkit's cell-8 whole-run lanes).
#[tokio::test]
async fn genesis_join_refuses_the_retired_cell6_adapter_and_routes_to_cell8() {
    let module = module_path();
    assert!(module.exists(), "tiny_llama_v2.wasm missing");
    let mut cfg = TrainClientConfig::new(env!("CARGO_BIN_EXE_daemon-vhc-worker").to_string());
    cfg.env = vec![
        (
            "DAEMON_TRAIN_MODULE".to_string(),
            module.to_string_lossy().into_owned(),
        ),
        // The owner's node-side lane choice (§9.6 numbers-are-config): CPU-admitting t2 lane.
        ("DAEMON_VHC_LANE_GPU_OPTIONAL".to_string(), "1".to_string()),
    ];
    cfg.spawn_timeout = Duration::from_secs(30);
    cfg.op_timeout = Duration::from_secs(180);
    let sup = TrainSupervisor::new(cfg);

    // Assess through the REAL funnel — unchanged by the retirement: the genesis worker role's
    // device_min feeds stage 3 and its grant list feeds stage 4.0 (D1 deliverable 4).
    let elig = sup.assess(genesis_wire()).await.expect("assess");
    assert!(
        elig.eligible,
        "v2 module still admits under the genesis role grants: {:?}",
        elig.reasons
    );

    // Join: the in-process native-coordinator drive for a genesis run no longer exists — the
    // typed refusal names the retirement and the superseding cell.
    let mut events = sup
        .join_streaming(RUN_LABEL, "local://native-tick", vec![], policy())
        .await
        .expect("join command accepted (the refusal arrives as the worker's typed error)");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let ev = tokio::time::timeout_at(deadline, events.recv())
            .await
            .expect("worker event stream stalled")
            .expect("worker event stream closed early");
        match ev {
            Event::Error { detail, .. } => {
                assert!(
                    detail.contains("cell 6 retired") && detail.contains("cell 8"),
                    "the refusal names the retirement and routes to the wasm coordinator: \
                     {detail}"
                );
                break;
            }
            Event::RoundOutcome { .. } => {
                panic!("a genesis run must NOT drive the retired native-coordinator session")
            }
            _ => {}
        }
    }
}
