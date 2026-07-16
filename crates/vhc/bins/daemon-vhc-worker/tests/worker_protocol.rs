// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The `daemon-vhc-worker` binary speaks the frozen `daemon_vhc_session::protocol` over the
// length-framed stdio cut, through `daemon-vhc-supervisor::TrainSupervisor` (probe / assess /
// join / throttle) against real guest modules.
//
// **Post-sunset shape (decisions D5)**: the v1 five-phase driver retired at the Phase-E sunset,
// so this suite's positive arms drive the REAL major-2 module (`tiny_llama_v2.wasm` — the v2
// whole-run itself is `tests/v2_join.rs`), and the suite carries the WIRE-LEVEL sunset
// regressions over the real binary:
//   * assessing the v1 `tiny_llama.wasm` returns an INELIGIBLE verdict with the typed
//     `AbiUnsupportedMajor` refusal code — an `Assessed` outcome, never an `Event::Error`, never
//     a crash (the protocol twin of the flipped A0 fixture);
//   * the D0 `UnsignedEnvelopeRetired` refusal is unchanged;
//   * `Throttle` never respawns the worker (preemption-as-churn is node-side post-sunset: the
//     arbiter pauses/stops and re-issues JoinRun on the durable intent).
//
// Dev/test harness: it shells `cargo build` for the guests and reads the `.wasm`, so the
// fs/process bans (which target the shipped node) are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{to_canonical_vec, Hash, SigningKey};
use daemon_vhc_sdk::models::TinyLlamaCfg;

use daemon_vhc_supervisor::{TrainClientConfig, TrainSupervisor};

// -- guest module loading (mirrors tests/v2_join.rs) ---------------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

/// RUSTFLAGS that make the guest `.wasm` byte-reproducible across checkouts/machines by remapping the
/// absolute prefixes rustc embeds in panic locations (the `<checkout>` root + the cargo registry).
/// MUST match `xtask build-guests` (`guest_remap_rustflags`) so a local rebuild reproduces the bytes
/// recorded in the committed `guests/guests.blake3`.
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

/// Stale-guest guard (Merge-1 adjudication): compare every module named in the committed
/// `guests/guests.blake3` against the `.wasm` in `dir`. A **missing / unreadable** module still
/// fails loud; a **hash mismatch** only WARNS (guest bytes are byte-reproducible within one
/// checkout, not across worktrees — see the Merge-1 decision in `docs/specs/swarm-p2-ledger.md`).
fn verify_guest_manifest(dir: &Path) {
    let manifest = guests_root().join("guests.blake3");
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
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            verify_guest_manifest(&guest_dir());
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
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

/// The tiny-llama-v2 guest config (`GuestCfg`) — the v2_join shape: single-peer roster, 2 inner
/// steps, one sequence per micro window.
fn v2_guest_config() -> Value {
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

/// A real signed schema-major-1 run envelope for `module` (resolved via `DAEMON_TRAIN_MODULE`
/// inside the signed path; the artifact entry is a placeholder the override substitutes).
fn signed_envelope_wire(run_id: &str, config: Value) -> Vec<u8> {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([1; 32]),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([2; 32]),
        },
    );
    let env = Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: run_id.to_string(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".to_string(),
            abi: "tensor-abi@1".to_string(),
            config,
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".to_string(),
            steps_per_round: 2,
            global_batch: GlobalBatch {
                start: 2,
                end: 2,
                ramp_rounds: 1,
            },
            stop: StopCondition::Rounds(4),
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
            warmup: 1,
            round_train_max: 60,
            round_witness: 1,
            cooldown: 1,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 2,
            payload_retention_rounds: 8,
        },
    };
    let author = SigningKey::from_bytes(&[0xA1u8; 32]);
    let frozen = env.freeze(&author).expect("freeze envelope");
    to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope")
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
/// implemented ABI major (2 since the sunset) and the full frozen 66-op `tabi@1` vocabulary (the
/// live §2.5 bridge surface — the sunset removed the DRIVER, not the vocabulary).
#[tokio::test]
async fn supervisor_probe_reports_major2_and_the_frozen_vocabulary() {
    let module = module_path("tiny_llama_v2.wasm");
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
    assert_eq!(
        hw.capabilities.abi_version, 2,
        "the implemented major is 2 post-sunset"
    );
    assert_eq!(
        hw.capabilities.ops.len(),
        66,
        "the host reports the full frozen tabi@1 vocabulary"
    );
    assert!(hw.capabilities.ops.iter().any(|o| o == "flash_attn@1"));
    sup.shutdown().await;
}

// -- THE SUNSET REGRESSION over the real binary (the flipped A0's protocol twin) ------------------

/// Assessing the pinned **v1** module post-sunset yields an INELIGIBLE `Assessed` verdict whose
/// refusal code is exactly `AbiUnsupportedMajor` — typed, attributable, never an `Event::Error`,
/// never a worker crash; the worker stays healthy and serves subsequent commands.
#[tokio::test]
async fn v1_module_assess_is_refused_abi_unsupported_major() {
    let module = module_path("tiny_llama.wasm");
    let sup = supervisor_for(&module);

    let cfg = TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    };
    let elig = sup
        .assess(signed_envelope_wire(
            "sunset-v1",
            Value::serialized(&cfg).expect("cfg value"),
        ))
        .await
        .expect("assess is an outcome, not a transport error");
    assert!(!elig.eligible, "a v1 module is refused post-sunset");
    assert_eq!(
        elig.refusal_code.as_deref(),
        Some("AbiUnsupportedMajor"),
        "the clean typed refusal (decisions D5), got: {:?}",
        elig.reasons
    );
    // The worker is unharmed: it still answers on the same process.
    sup.ping().await.expect("worker healthy after the refusal");
    assert_eq!(sup.restarts().await, 0, "no respawn");
    sup.shutdown().await;
}

// -- the v2 positive over the signed-envelope seam ------------------------------------------------

/// The Merge-3 envelope seam, post-sunset: the worker receives the real signed envelope, verifies
/// it, and the **major-2** module assesses eligible through the claim funnel (the full v2
/// whole-run join is `tests/v2_join.rs`).
#[tokio::test]
async fn v2_module_assesses_eligible_over_the_signed_envelope() {
    let module = module_path("tiny_llama_v2.wasm");
    let sup = supervisor_for(&module);

    let elig = sup
        .assess(signed_envelope_wire("worker-seam", v2_guest_config()))
        .await
        .expect("assess over the signed envelope");
    assert!(
        elig.eligible,
        "the v2 module assesses eligible: {:?}",
        elig.reasons
    );
    sup.shutdown().await;
}

/// D0: the unsigned legacy envelope path is RETIRED with a typed refusal (refactor §8/D0). Raw
/// config CBOR (no `SignedEnvelope` wrapper) — the pre-A0 direct-drive — is refused with the
/// stable `UnsignedEnvelopeRetired` slug even when `DAEMON_TRAIN_MODULE` is set.
#[tokio::test]
async fn unsigned_raw_config_assess_is_refused_with_typed_slug() {
    let module = module_path("tiny_llama_v2.wasm");
    let sup = supervisor_for(&module);

    let raw = to_canonical_vec(&v2_guest_config()).expect("raw config cbor");
    let err = sup
        .assess(raw)
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
    let module = module_path("tiny_llama_v2.wasm");
    let sup = supervisor_for(&module);

    sup.assess(signed_envelope_wire("run-9", v2_guest_config()))
        .await
        .expect("assess");
    sup.throttle(None, None, true).await.expect("pause");
    sup.throttle(None, None, false).await.expect("resume");
    sup.assess(signed_envelope_wire("run-9", v2_guest_config()))
        .await
        .expect("assess after the lever");
    assert_eq!(
        sup.restarts().await,
        0,
        "the throttle lever is churn-free on the worker process"
    );
    sup.shutdown().await;
}
