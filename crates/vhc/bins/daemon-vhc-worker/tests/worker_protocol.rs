// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The `daemon-vhc-worker` binary speaks the frozen `daemon_vhc_session::protocol` over the
// (Run descriptions are genesis envelopes v2 since the worker's join re-seated onto the wasm
// coordinator; the schema-major-1 envelope form refuses typed at assess — pinned below.)
// length-framed stdio cut, through `daemon-vhc-supervisor::TrainSupervisor` (probe / assess /
// join / throttle) against real guest modules.
//
// **Post-sunset shape (decisions D5)**: the v1 five-phase driver retired at the Phase-E sunset,
// so this suite's positive arms drive the REAL major-2 module (`tiny_llama.wasm` — the v2
// whole-run itself is `tests/join.rs`), and the suite carries the WIRE-LEVEL sunset
// regressions over the real binary:
//   * assessing a SYNTHETIC ABI-major-1 module (a few-section wasm image hand-assembled in-test
//     with `wasm-encoder`: empty imports + the v1 lifecycle exports + `da_abi` = major 1 — no
//     vendored/recorded bytes) returns an INELIGIBLE verdict with the typed `AbiUnsupportedMajor`
//     refusal code — an `Assessed` outcome, never an `Event::Error`, never a crash (the protocol
//     twin of the host-side driver-selection refusal, `daemon-vhc-host/tests/driver_selection.rs`);
//   * the D0 `UnsignedEnvelopeRetired` refusal is unchanged;
//   * `Throttle` never respawns the worker (preemption-as-churn is node-side post-sunset: the
//     arbiter pauses/stops and re-issues JoinRun on the durable intent).
//
// Dev/test harness: it shells `cargo build` for the guests and reads the `.wasm`, so the
// fs/process bans (which target the shipped node) are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;
use daemon_vhc_proto::envelope::Access;
use daemon_vhc_proto::{peer_id, to_canonical_vec, Hash, PeerId, SigningKey};

use daemon_vhc_supervisor::{TrainClientConfig, TrainSupervisor};

// -- guest module loading (mirrors tests/join.rs) ---------------------------------------------

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("VHC_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    daemon_vhc_guest_build::guests_root().join("target/wasm32-unknown-unknown/release")
}

/// Stale-guest guard (Merge-1 adjudication): compare every module named in the committed
/// `guests/guests.blake3` against the `.wasm` in `dir`. A **missing / unreadable** module still
/// fails loud; a **hash mismatch** only WARNS (guest bytes are byte-reproducible within one
/// checkout, not across worktrees — see the Merge-1 decision in `docs/archive/swarm-p2-ledger.md`).
fn verify_guest_manifest(dir: &Path) {
    let manifest = daemon_vhc_guest_build::guests_root().join("guests.blake3");
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
        panic!(
            "read guest manifest {}: {e} — run `cargo run -p xtask -- build-guests`",
            manifest.display()
        )
    });
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (hex, name) = line
            .split_once("  ")
            .expect("guests.blake3 line must be `<blake3-hex>  <name>.wasm`");
        let bytes = std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("read guest module {}/{name}: {e}", dir.display()));
        let got = blake3::hash(&bytes).to_hex();
        if got.as_str() != hex {
            eprintln!(
                "warning: guest `{name}` in {} hashes {got} but committed guests.blake3 records \
                 {hex}. This is expected across worktrees/machines (path-keyed codegen ordering, \
                 not a stale artifact); the freshly-built module is used. If you changed guest \
                 source, run `cargo run -p xtask -- build-guests` and commit guests/guests.blake3.",
                dir.display()
            );
        }
    }
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("VHC_TEST_GUEST_DIR").is_ok() {
            verify_guest_manifest(&guest_dir());
            return;
        }
        daemon_vhc_guest_build::ensure_built().unwrap_or_else(|e| panic!("{e}"));
        verify_guest_manifest(&guest_dir());
    });
}

fn module_path(name: &str) -> PathBuf {
    let path = guest_dir().join(name);
    if !path.exists() {
        ensure_built();
    }
    assert!(path.exists(), "{name} missing at {}", path.display());
    path
}

fn worker_bin() -> String {
    env!("CARGO_BIN_EXE_daemon-vhc-worker").to_string()
}

// -- synthetic ABI-major-1 module (the AbiUnsupportedMajor refusal input, constructed in-test) ----

/// The v1 lifecycle export set — a candidate-major-1 import/export shape (∅ imports ⊆ {tabi@1}).
const V1_LIFECYCLE_EXPORTS: &[&str] = &[
    "da_alloc",
    "da_free",
    "da_manifest",
    "da_build",
    "da_step",
    "da_inner_update",
    "da_make_update",
    "da_ingest_updates",
];

/// Assemble a minimal valid wasm module declaring ABI major 1: empty imports, the v1 lifecycle
/// exports (all `() -> i32`), plus a `da_abi` export returning `pack(1, 0)`. This is the offending
/// input the post-sunset host refuses with a typed `AbiUnsupportedMajor` — hand-built here (the
/// same `wasm-encoder` shape `daemon-vhc-host/tests/driver_selection.rs` uses) so no vendored,
/// recorded pre-refactor artifact is load-bearing for the refusal proof.
fn synthetic_v1_module() -> Vec<u8> {
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Module, TypeSection,
        ValType,
    };
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I32]);
    module.section(&types);

    let n_funcs = V1_LIFECYCLE_EXPORTS.len() as u32 + 1; // + da_abi
    let mut funcs = FunctionSection::new();
    for _ in 0..n_funcs {
        funcs.function(0);
    }
    module.section(&funcs);

    let mut exports = ExportSection::new();
    for (i, name) in V1_LIFECYCLE_EXPORTS.iter().enumerate() {
        exports.export(name, ExportKind::Func, i as u32);
    }
    exports.export(
        "da_abi",
        ExportKind::Func,
        V1_LIFECYCLE_EXPORTS.len() as u32,
    );
    module.section(&exports);

    let mut code = CodeSection::new();
    for _ in 0..V1_LIFECYCLE_EXPORTS.len() {
        let mut f = Function::new([]);
        f.instructions().i32_const(0).end();
        code.function(&f);
    }
    let mut da_abi = Function::new([]);
    da_abi.instructions().i32_const(1 << 16).end(); // pack(major=1, minor=0)
    code.function(&da_abi);
    module.section(&code);

    module.finish()
}

/// Write bytes to a unique temp `.wasm` path (the worker resolves its module from a filesystem
/// path via `DAEMON_TRAIN_MODULE`). Returns the path; the caller removes it after the assess.
fn write_temp_module(bytes: &[u8], tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "daemon-vhc-{tag}-{}-{nanos}.wasm",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write synthetic module to temp");
    path
}

/// The compute@2 trainer's canonical parameter element counts for the tiny t2 parity shape
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

/// The compute@2 trainer's guest config (`GuestCfg`, authored SDK-free as raw canonical CBOR —
/// the tiny t2 parity model + `sparse_loco` profile + matched init): single-peer roster, 2 inner
/// steps, one sequence per micro window.
fn guest_config() -> Value {
    let model = Value::Map(vec![
        (Value::from("d_model"), Value::from(64u32)),
        (Value::from("n_layers"), Value::from(1u32)),
        (Value::from("n_heads"), Value::from(4u32)),
        (Value::from("head_dim"), Value::from(16u32)),
        (Value::from("vocab"), Value::from(64u32)),
        (Value::from("seq_len"), Value::from(9u32)),
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
        (Value::from("profile"), profile),
        (
            Value::from("state"),
            Value::serialized(&seed_state_contract()).expect("state contract"),
        ),
    ])
}

/// The genesis seed-form state contract (§6.1a) for this suite's trainer layout — replaces the
/// deleted inline init (the guest seals the seed expansion and cross-checks `expected_root`).
fn seed_state_contract() -> daemon_vhc_proto::genesis::StateContract {
    use daemon_vhc_proto::det_state::{derive_state_chunk_size, FamilyEntry};
    use daemon_vhc_proto::genesis::{StateContract, StateInit};
    let seed = [0x5eu8; 32];
    let dist = daemon_vhc_det::SEED_INIT_DIST_V1;
    let chunk_size = derive_state_chunk_size(64);
    let param_bytes: Vec<Vec<u8>> = param_numels()
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let vals =
                daemon_vhc_det::seed_init_param(&seed, dist, i as u64, n).expect("known dist");
            let mut b = Vec::with_capacity(n * 4);
            for v in vals {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        })
        .collect();
    let views: Vec<&[u8]> = param_bytes.iter().map(Vec::as_slice).collect();
    let expected_root = FamilyEntry::author(&views, chunk_size)
        .expect("author")
        .fold;
    StateContract {
        chunk_size,
        init: StateInit::Seed {
            seed: daemon_vhc_proto::Seed(seed),
            dist,
            expected_root,
        },
    }
}

/// A real signed **genesis envelope v2** wire for the run (the worker's only resolvable form):
/// a coordinator role + a trainer role carrying `config`, with placeholder artifact entries the
/// `DAEMON_TRAIN_MODULE` override substitutes (the explicit dev/node-controlled module source
/// inside the signed path — assess never fetches the coordinator module).
fn genesis_wire(run_label: &str, config: Value) -> Vec<u8> {
    genesis_wire_seated(run_label, config, None)
}

/// [`genesis_wire`] with an optional SEAT BINDING on the trainer role (defect 6: an
/// identity-bound role authored for one participant's base identity).
fn genesis_wire_seated(run_label: &str, config: Value, seat: Option<PeerId>) -> Vec<u8> {
    use daemon_vhc_proto::genesis::{
        ChannelDecl, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
        TransportSelection, GENESIS_SCHEMA_MAJOR,
    };
    use daemon_vhc_proto::GenesisEnvelope;
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "worker-mod".to_string(),
        SnapshotArtifact {
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
            // A fixture envelope: this exercises paths that have nothing to do with resources, and it
            // uses the SAME shared trivial construction every compute-free module emits.
            execution: Some(
                daemon_vhc_proto::RoleExecutionRequirements::fixture_over_trivial_plan(vec![
                    "cpu".to_string()
                ]),
            ),
            lane: "trainer".into(),
            module: "worker-mod".into(),
            abi: "vhc@2".into(),
            config,
            // The control channel the module's manifest names (§9.4 step 6: an envelope that
            // omitted channel 0 would be a typed GrantsExceedLane refusal); numeric quotas
            // inherit the lane ceiling (tighten-only).
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
            // The superseded device-minimums section stays EMPTY: physical requirements are
            // members of the composed estimate, and an authored minimum beside a composed one is a
            // second authority over the same question, which authoring refuses.
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
            identity: seat,
        },
    );
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
            // A fixture envelope: this exercises paths that have nothing to do with resources, and it
            // uses the SAME shared trivial construction every compute-free module emits.
            execution: Some(
                daemon_vhc_proto::RoleExecutionRequirements::fixture_over_trivial_plan(vec![
                    "cpu".to_string()
                ]),
            ),
            lane: "coordinator".into(),
            module: "coord-mod".into(),
            abi: "vhc@2".into(),
            config: Value::Map(vec![]),
            grants: RoleGrants::default(),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
            identity: None,
        },
    );
    let env = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: run_label.to_string(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        state_contract: Some(seed_state_contract()),
        // The opaque Authority section (D1 vocabulary; the host never interprets it) — nominal
        // for assess-only drives.
        authority: Value::Map(vec![
            (Value::from("topology"), Value::from("single-key")),
            (Value::from("coordinator"), Value::Bytes(vec![9u8; 32])),
        ]),
        transport: TransportSelection::default(),
        identities: Identities::default(),
    };
    let author = SigningKey::from_bytes(&[0xA1u8; 32]);
    let frozen = env.freeze(&author).expect("freeze genesis");
    let wire = daemon_vhc_proto::SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    to_canonical_vec(&wire).expect("wire")
}

/// A signed wire whose payload is a SYNTHETIC schema-major-1 envelope — the RETIRED v1 form,
/// authored here only as the typed-refusal pin's input (`EnvelopeSchemaRetired` at assess). The
/// refusal is decided by the outer `[run].schema` read alone, before any signature or payload
/// consideration, so the pin needs no retired v1 payload machinery: a canonical-CBOR map carrying
/// `[run].schema = 1` inside the `SignedEnvelope` transport wrapper is the complete input.
fn signed_envelope_wire(run_id: &str) -> Vec<u8> {
    let run = Value::Map(vec![
        (Value::from("schema"), Value::from(1u32)),
        (Value::from("run_id"), Value::from(run_id)),
    ]);
    let envelope = Value::Map(vec![(Value::from("run"), run)]);
    let wire = daemon_vhc_proto::SignedEnvelope {
        bytes: to_canonical_vec(&envelope).expect("envelope cbor"),
        signature: daemon_vhc_proto::Signature([0u8; 64]),
        signer: daemon_vhc_proto::peer_id(&SigningKey::from_bytes(&[0xA1u8; 32])),
    };
    to_canonical_vec(&wire).expect("encode signed envelope")
}

fn supervisor_for(module: &Path) -> TrainSupervisor {
    let mut cfg = TrainClientConfig::new(worker_bin());
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
    TrainSupervisor::new(cfg)
}

// -- probe: the frozen capability report ----------------------------------------------------------

/// CLI-1 / RUN-9 worker side: the supervisor spawns the real worker; the probe reports the
/// implemented ABI major (2 — the only major) and the major-2 capability vocabulary: the five
/// worlds at their implemented minor plus the versioned custom-op registry. No `tabi@1` entry
/// exists — the retired bridge is a typed refusal, not a capability.
#[tokio::test]
async fn supervisor_probe_reports_major2_worlds_and_custom_ops() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    let hw = sup.probe().await.expect("probe");
    if cfg!(any(feature = "wgpu", feature = "cuda")) {
        assert!(hw.gpus <= 1, "the GPU probe reports 0 or 1 usable devices");
        assert!(
            hw.backend_lanes.iter().any(|l| l == "cpu"),
            "the cpu lane is always present"
        );
    } else {
        assert_eq!(hw.gpus, 0, "this build has no GPU lane");
    }
    assert_eq!(hw.capabilities.abi_version, 2, "the implemented major is 2");
    for world in ["vhc@2", "net@2", "sys@2", "data@2", "compute@2"] {
        assert!(
            hw.capabilities
                .ops
                .iter()
                .any(|o| o.starts_with(&format!("{world}:"))),
            "the probe advertises {world} with its implemented minor: {:?}",
            hw.capabilities.ops
        );
    }
    assert!(
        hw.capabilities.ops.iter().any(|o| o == "flash_attn@1"),
        "the custom-op registry is advertised"
    );
    assert!(
        !hw.capabilities.ops.iter().any(|o| o.contains("tabi")),
        "no retired-bridge vocabulary is advertised"
    );
    sup.shutdown().await;
}

/// WS6 regression (the crash-loop meltdown fix): the shipped `[vhc]` config default names the
/// REAL worker binary, and a binary of that name speaks `Event::Ready` when the supervisor spawns
/// it. Before the v1-retirement the default was `"daemon-vhc"` — the initial scaffold that printed a
/// version line and exited, so a stock `[vhc] enabled` node crash-looped its supervisor on spawn
/// (the negative half of this contract is the `supervisor_meltdown` unit test over a bogus path).
/// `Worker::spawn` blocks on `Event::Ready`, so a successful `probe()` here proves the configured
/// default resolves to a `Ready`-speaking worker, not a self-exiting stub.
#[tokio::test]
async fn configured_default_worker_path_names_a_binary_that_speaks_ready() {
    // The product default the node feeds to `TrainClientConfig::new` (bins/daemon/src/main.rs).
    let default_path = daemon_vhc_session::config::VhcConfig::default().worker_path;
    assert_eq!(
        default_path, "daemon-vhc-worker",
        "the shipped default must name the real worker binary, not the retired `daemon-vhc` scaffold"
    );
    // The binary cargo built for this `[[bin]]` target — its file name is exactly the default the
    // node would look up on `PATH`, tying the config contract to a binary that actually exists.
    let bin = worker_bin();
    assert_eq!(
        Path::new(&bin).file_name().and_then(|s| s.to_str()),
        Some(default_path.as_str()),
        "the built worker binary's name must equal the configured `worker_path` default"
    );

    // Spawn it exactly as the node would; the supervisor blocks until the worker reports `Ready`.
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);
    sup.probe()
        .await
        .expect("the configured default worker reaches Ready and answers a probe");
    assert_eq!(
        sup.restarts().await,
        0,
        "a healthy default never triggers a respawn"
    );
    sup.shutdown().await;
}

// -- THE SUNSET REGRESSION over the real binary (the flipped A0's protocol twin) ------------------

/// Assessing a SYNTHETIC ABI-major-1 module post-sunset yields an INELIGIBLE `Assessed` verdict
/// whose refusal code is exactly `AbiUnsupportedMajor` — typed, attributable, never an
/// `Event::Error`, never a worker crash; the worker stays healthy and serves subsequent commands.
/// The offending module is hand-assembled in-test (no vendored/recorded pre-refactor bytes).
#[tokio::test]
async fn v1_module_assess_is_refused_abi_unsupported_major() {
    let module = write_temp_module(&synthetic_v1_module(), "synthetic-v1-assess");
    let sup = supervisor_for(&module);

    let elig = sup
        .assess(genesis_wire("abi-major-1", guest_config()), None)
        .await
        .expect("assess is an outcome, not a transport error");
    assert!(!elig.eligible, "a major-1 module is refused post-sunset");
    assert_eq!(
        elig.refusal_code.as_deref(),
        Some("AbiUnsupportedMajor"),
        "the clean typed refusal (decisions D5), got: {:?}",
        elig.reasons
    );
    // The worker is unharmed: it still answers on the same process (no partial admission, no
    // guest code executed — the refusal is an admission outcome raised before any da_* dispatch).
    sup.ping().await.expect("worker healthy after the refusal");
    assert_eq!(sup.restarts().await, 0, "no respawn");
    sup.shutdown().await;
    let _ = std::fs::remove_file(&module);
}

// -- the certification-minor truth over the signed-envelope seam ---------------------------------

/// The envelope seam over the genesis form for a **certification-minor** module: the worker
/// verifies the real signed genesis, the trainer emits its Logical Resource Plan — and the worker
/// refuses `EstimateNotComposable`, TYPED and healthy, because no authenticated Backend Execution
/// Profile is provisioned on this box to compose a physical estimate with. That refusal is the
/// correct answer today, not a gap: composing without an authenticated profile would be the exact
/// substitution the resource model exists to prevent. When node-side profile provisioning lands
/// (the measurement wave's deliverable), this seam flips eligible and this test flips with it.
/// The declared-claim positive over the same seam lives in `tests/seat_smoke.rs`.
#[tokio::test]
async fn certification_minor_assess_refuses_claim_not_composable_without_a_profile() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    let elig = sup
        .assess(genesis_wire("worker-seam", guest_config()), None)
        .await
        .expect("assess over the signed genesis");
    assert!(
        !elig.eligible,
        "no authenticated profile is provisioned, so a certification-minor module must refuse"
    );
    assert!(
        elig.reasons
            .iter()
            .any(|r| r.contains("EstimateNotComposable")),
        "the refusal carries the stable typed slug: {:?}",
        elig.reasons
    );
    sup.ping().await.expect("worker healthy after the refusal");
    assert_eq!(sup.restarts().await, 0, "no respawn");
    sup.shutdown().await;
}

/// The SEATED role set refuses an undirected assess (defect 6 of the c15 drills): an
/// identity-bound trainer role carries one seat's frozen plan identity, and the worker's
/// map-order default would hand EVERY undirected joiner the same first seat — both boxes
/// training the same window slice, the checkpoint slots elected to the other seat published by
/// nobody. Seat selection belongs to the node (it holds the identity keystore and picks the
/// role whose binding matches its base identity); the worker's part of the contract is to
/// refuse to guess. Directed to the seat by name, the same wire proceeds past selection (into
/// this suite's standing no-profile refusal — proof the refusal above is the selection's own).
#[tokio::test]
async fn seated_genesis_refuses_an_undirected_assess() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    let seat = peer_id(&SigningKey::from_bytes(&[0x5Au8; 32]));
    let wire = genesis_wire_seated("seated-run", guest_config(), Some(seat));

    let err = sup
        .assess(wire.clone(), None)
        .await
        .expect_err("an undirected assess against a seated role set must refuse");
    assert!(
        err.to_string().contains("identity-bound seats"),
        "the refusal names the seat contract, got: {err}"
    );

    // Directed to the authored seat, selection succeeds — the run proceeds to the standing
    // typed no-profile refusal (`EstimateNotComposable`), which is PAST role selection.
    let elig = sup
        .assess(wire, Some("trainer".to_string()))
        .await
        .expect("a directed assess is an outcome, not a transport error");
    assert!(
        elig.reasons
            .iter()
            .any(|r| r.contains("EstimateNotComposable")),
        "the directed assess reached the resource seam: {:?}",
        elig.reasons
    );

    sup.ping().await.expect("worker healthy after the refusal");
    assert_eq!(sup.restarts().await, 0, "no respawn");
    sup.shutdown().await;
}

/// The retired schema-major-1 envelope form refuses TYPED at assess (`EnvelopeSchemaRetired`):
/// a v1 envelope cannot configure a coordinator (no coordinator role entry, no
/// Authority/identities section), so the worker refuses it before any module is fetched — and
/// stays healthy on the same process.
#[tokio::test]
async fn v1_envelope_assess_is_refused_envelope_schema_retired() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    let err = sup
        .assess(signed_envelope_wire("retired-v1-form"), None)
        .await
        .expect_err("a schema-1 envelope must refuse at assess");
    assert!(
        err.to_string().contains("EnvelopeSchemaRetired"),
        "the refusal carries the stable typed slug, got: {err}"
    );
    sup.ping().await.expect("worker healthy after the refusal");
    assert_eq!(sup.restarts().await, 0, "no respawn");
    sup.shutdown().await;
}

/// D0: the unsigned legacy envelope path is RETIRED with a typed refusal (refactor §8/D0). Raw
/// config CBOR (no `SignedEnvelope` wrapper) — the pre-A0 direct-drive — is refused with the
/// stable `UnsignedEnvelopeRetired` slug even when `DAEMON_TRAIN_MODULE` is set.
#[tokio::test]
async fn unsigned_raw_config_assess_is_refused_with_typed_slug() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    let raw = to_canonical_vec(&guest_config()).expect("raw config cbor");
    let err = sup
        .assess(raw, None)
        .await
        .expect_err("raw-config assess must refuse (retired at D0)");
    assert!(
        err.to_string().contains("UnsignedEnvelopeRetired"),
        "the refusal carries the stable typed slug, got: {err}"
    );
    sup.shutdown().await;
}

/// RUN-9 (§10.5) post-sunset: `Throttle` never harms or respawns the worker — preemption-as-churn
/// moved node-side (the arbiter stops the instance and re-issues JoinRun on the durable intent);
/// the worker survives the lever and serves a fresh assess on the same process.
#[tokio::test]
async fn throttle_is_harmless_and_never_respawns() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    sup.assess(genesis_wire("run-9", guest_config()), None)
        .await
        .expect("assess");
    sup.throttle(None, None, true).await.expect("pause");
    sup.throttle(None, None, false).await.expect("resume");
    sup.assess(genesis_wire("run-9", guest_config()), None)
        .await
        .expect("assess after the lever");
    assert_eq!(
        sup.restarts().await,
        0,
        "the throttle lever is churn-free on the worker process"
    );
    sup.shutdown().await;
}
