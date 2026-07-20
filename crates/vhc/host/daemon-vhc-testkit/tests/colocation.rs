// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// THE Phase-E colocation acceptance over PRODUCTION blobs (refactor §9: "trainer + verifier
// colocated on one host, arbitrated, both green"; decisions D6), tier-2:
//
// Two real wasm role-instances run CONCURRENTLY in one host process — a trainer
// (`tiny_llama.wasm`, the end-state barrier whole-run under the PRODUCTION `coordinator_quorum.wasm`
// coordinator) and a verifier-role instance (`toy_averager.wasm`, self-driven) — each admitted
// through the node's
// `OwnerArbiter` (per-device + host-wide typed ledgers, atomic check-and-reserve) BEFORE its
// sandbox starts, exactly the D6 funnel order. Both runs end green (clean outcome + §8.7 replay
// bit-for-bit); the ledgers account exactly while both are live; an over-budget third instance
// is refused TYPED while they run; and the releases happen on observed teardown (after each
// run's report — the sandbox has stopped), restoring the full budget.
//
// The role label is the envelope-level string the host never interprets (decisions D1): the
// verifier-lane instance runs a distinct module under the "verifier" role — role enablement is
// config, not re-architecture (decisions D7 consequence for E).
//
// Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Once};
use std::time::Duration;

use daemon_vhc_host::run::{RunEnd, RunIdentity};
use daemon_vhc_node::{AdmitRefusal, InstanceCharge, OwnerArbiter, OwnerBudget, RoleInstanceId};
use daemon_vhc_testkit::run::{whole_run, RunSpec};
use daemon_vhc_testkit::{genesis_whole_run, GenesisRunSpec};

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

fn guest(name: &str) -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env_remove("RUSTC_WRAPPER")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

const GIB: u64 = 1 << 30;

fn id(label: &str, role: &str, instance: u64) -> RoleInstanceId {
    RoleInstanceId {
        run_id: *blake3::hash(label.as_bytes()).as_bytes(),
        epoch: 0,
        role: role.to_string(),
        instance,
    }
}

#[test]
fn trainer_and_verifier_colocated_on_one_host_arbitrated_both_green() {
    let coordinator_wasm = guest("coordinator_quorum");
    let trainer_wasm = guest("tiny_llama");
    let verifier_wasm = guest("toy_averager");

    // The owner's aggregate grants: one 8 GiB accelerator, 100% duty, at most 2 instances.
    let arbiter = Arc::new(OwnerArbiter::new(OwnerBudget {
        device_memory: BTreeMap::from([("gpu:0".to_string(), 8 * GIB)]),
        host_ram: 64 * GIB,
        disk: u64::MAX,
        net_up_bps: u64::MAX,
        net_down_bps: u64::MAX,
        duty_pct: 100,
        max_instances: 2,
    }));

    // Admission BEFORE any sandbox starts (the D6 funnel's last stage, atomically reserved).
    let trainer_id = id("colo-trainer", "trainer", 1);
    let verifier_id = id("colo-verifier", "verifier", 2);
    arbiter
        .admit(
            trainer_id.clone(),
            InstanceCharge::device_memory("gpu:0", 5 * GIB, 50),
            100,
        )
        .expect("trainer admitted");
    arbiter
        .admit(
            verifier_id.clone(),
            InstanceCharge::device_memory("gpu:0", 2 * GIB, 30),
            100,
        )
        .expect("verifier colocated");
    // The ledgers account exactly while both instances are live.
    let snap = arbiter.remaining();
    assert_eq!(snap.device_memory["gpu:0"], GIB);
    assert_eq!(snap.duty_pct, 20);
    assert_eq!(snap.instances, 2);

    // While both are live, an over-budget third instance is refused TYPED — never optimistic.
    assert!(matches!(
        arbiter.admit(
            id("colo-third", "trainer", 3),
            InstanceCharge::device_memory("gpu:0", 2 * GIB, 10),
            100,
        ),
        Err(AdmitRefusal::MaxInstances { max: 2 })
    ));

    // Run both role-instances CONCURRENTLY — two sandboxes, one host process.
    let trainer = std::thread::spawn({
        let coordinator = coordinator_wasm;
        let wasm = trainer_wasm;
        move || {
            let spec = GenesisRunSpec::new("colo-trainer", 1, 2);
            genesis_whole_run(&coordinator, &wasm, &spec).expect("trainer whole run")
        }
    });
    let verifier = std::thread::spawn({
        let wasm = verifier_wasm.clone();
        move || {
            let worker = daemon_vhc_testkit::worker().expect("testkit worker");
            let identity = RunIdentity {
                run_id: *blake3::hash(b"colo-verifier").as_bytes(),
                epoch: 0,
                role: "verifier".to_string(),
                instance: 2,
                module: *blake3::hash(&verifier_wasm).as_bytes(),
            };
            let spec = RunSpec {
                timeout: Duration::from_secs(60),
                ..RunSpec::self_driven(identity, [0x5C; 32], vec![3u8], Vec::new(), 3)
            };
            whole_run(&worker, &wasm, spec).expect("verifier whole run")
        }
    });

    let trainer_report = trainer.join().expect("trainer thread");
    let verifier_report = verifier.join().expect("verifier thread");

    // Both green: clean outcomes + §8.7 replays bit-for-bit (+ the trainer's det-lane digest
    // agreement across its roster).
    assert!(trainer_report.is_green(), "trainer green under colocation");
    assert!(
        matches!(verifier_report.end, RunEnd::Outcome(0)),
        "verifier clean outcome, got {:?}",
        verifier_report.end
    );
    assert!(
        verifier_report.is_green(),
        "verifier green under colocation"
    );

    // Observed teardown (both sandboxes have stopped — the reports are their terminal facts) →
    // release → the full budget returns and the third instance now admits.
    assert!(arbiter.release(&trainer_id));
    assert!(arbiter.release(&verifier_id));
    let snap = arbiter.remaining();
    assert_eq!(snap.device_memory["gpu:0"], 8 * GIB);
    assert_eq!(snap.instances, 0);
    arbiter
        .admit(
            id("colo-third", "trainer", 3),
            InstanceCharge::device_memory("gpu:0", 2 * GIB, 10),
            100,
        )
        .expect("admits after observed teardown");
}
