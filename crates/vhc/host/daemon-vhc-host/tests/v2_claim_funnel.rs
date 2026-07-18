// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The A2 claim + admission-funnel acceptance (refactor §5 A2; §10 gate table row "Claim rejection
// / over-claim / under-claim traps", tier-1; ABI §9 conformance rows): the `test-claim-v2` guest
// drives every lane through the REAL restricted assessment instance and funnel —
//
//   - over-claim rejected against owner policy (stage 5, ClaimExceedsPolicy);
//   - claim outside the lane's claim bounds refused (stage 4, ClaimExceedsPolicy);
//   - a nondeterministic claim refused (ClaimInconsistent);
//   - a manifest naming an unadmitted channel refused (GrantsExceedLane);
//   - an under-claimer trapping ATTRIBUTABLY at its own hard-accountable cap at run time;
//   - determinism: same (config, grants) → byte-identical claim across whole assessments.
//
// The v1 path's typed refusal (AbiUnsupportedMajor since the Phase-E sunset) is pinned by the A0
// frozen-fixture lane, not re-tested here.
//
// Dev/test harness: shells `cargo build` for the guests (same pattern as worker_protocol.rs).
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};

use daemon_vhc_abi::AbiRefusalCode;
use daemon_vhc_host::v2::{
    admit_v2, start_run, DeviceProfile, MemorySink, OwnerPolicy, ParticipationLane, RunEnd,
    RunIdentity, V2RunConfig,
};
use daemon_vhc_host::{EngineConfig, TrapCode, Worker};

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

fn test_claim_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/test_claim_v2.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// A permissive test lane (numbers are deployment config — §9.6): GPU optional, no floors,
/// 4 GiB claim bounds per tier.
fn lane() -> ParticipationLane {
    ParticipationLane {
        lane: "trainer".into(),
        version: 1,
        enabled: true,
        gpu: 1,
        vram_bytes: 0,
        ram_bytes: 0,
        disk_bytes: 0,
        claim_bounds_device: [0, 4 << 30],
        claim_bounds_host: [0, 4 << 30],
        ceilings: ParticipationLane::trainer_launch_defaults().ceilings,
    }
}

fn device() -> DeviceProfile {
    DeviceProfile {
        gpu: false,
        vram_bytes: 0,
        ram_bytes: 16 << 30,
        disk_bytes: 100 << 30,
    }
}

fn owner_uncapped() -> OwnerPolicy {
    OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    }
}

fn worker() -> Worker {
    Worker::new(EngineConfig::default()).expect("engine")
}

fn identity(wasm: &[u8], instance: u64) -> RunIdentity {
    RunIdentity {
        run_id: [0xCC; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(wasm).as_bytes(),
    }
}

// -- positive: the honest claimer admits, runs, and its claim bytes reach the run header ----------

#[test]
fn honest_claim_admits_and_runs_with_deterministic_claim_bytes() {
    let wasm = test_claim_wasm();
    let w = worker();
    let cfg = vec![0u8, 0u8]; // mode 0

    let admission = admit_v2(
        &w,
        &wasm,
        Some(blake3::hash(&wasm).as_bytes()),
        &cfg,
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        None,
        None,
    )
    .expect("honest claim admits");
    assert_eq!(admission.claim.hard_accountable.host, 1 << 16);
    assert_eq!(admission.claim.under_pressure, vec![0, 1]);
    assert!(!admission.claim_bytes.is_empty());
    assert!(!admission.manifest_bytes.is_empty());

    // Determinism (§9.2): a whole second assessment yields byte-identical claim bytes.
    let again = admit_v2(
        &w,
        &wasm,
        None,
        &cfg,
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        None,
        None,
    )
    .expect("second assessment");
    assert_eq!(admission.claim_bytes, again.claim_bytes);

    // The admitted claim wires into the run: header bytes + the enforced hard cap.
    let mut run_cfg = V2RunConfig::new(identity(&wasm, 1), [0x61; 32], cfg, Vec::new());
    run_cfg.claim_bytes = admission.claim_bytes.clone();
    run_cfg.manifest_bytes = admission.manifest_bytes.clone();
    run_cfg.hard_accountable_host_bytes = admission.claim.hard_accountable.host;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&w, &wasm, run_cfg, Box::new(sink)).expect("start");
    run.pump
        .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("join") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }
}

// -- over-claim vs OWNER policy (stage 5): the funnel's supreme last stage -------------------------

#[test]
fn over_claim_rejected_against_owner_policy() {
    let wasm = test_claim_wasm();
    // Mode 4 param 2: the module claims 2 GiB device; the lane allows it (4 GiB bounds) but the
    // owner's standing cap is 1 GiB — stage 5 refuses, owner supreme.
    let owner = OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 1 << 30,
        host_cap_bytes: 0,
    };
    let err = admit_v2(
        &worker(),
        &wasm,
        None,
        &[4u8, 2u8],
        &[],
        &lane(),
        &device(),
        &owner,
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(err.stage, 5);
    assert_eq!(err.code, Some(AbiRefusalCode::ClaimExceedsPolicy));
    assert!(err.reason.contains("owner"), "{}", err.reason);
}

// -- claim outside the LANE's sanity bounds (stage 4 tail) ------------------------------------------

#[test]
fn claim_outside_lane_bounds_refused() {
    let wasm = test_claim_wasm();
    // Mode 4 param 8: 8 GiB device claim > the lane's 4 GiB bound — refused at stage 4 (before
    // any owner judgment), also ClaimExceedsPolicy but naming the lane.
    let err = admit_v2(
        &worker(),
        &wasm,
        None,
        &[4u8, 8u8],
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(err.stage, 4);
    assert_eq!(err.code, Some(AbiRefusalCode::ClaimExceedsPolicy));
    assert!(err.reason.contains("lane"), "{}", err.reason);
}

// -- a nondeterministic claim (§9.2 byte-identity) ---------------------------------------------------

#[test]
fn inconsistent_claim_refused() {
    let wasm = test_claim_wasm();
    let err = admit_v2(
        &worker(),
        &wasm,
        None,
        &[1u8, 0u8],
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(err.stage, 4);
    assert_eq!(err.code, Some(AbiRefusalCode::ClaimInconsistent));
}

// -- manifest beyond the admitted channel table (§9.4 step 6) ----------------------------------------

#[test]
fn manifest_channel_beyond_table_refused_grants_exceed_lane() {
    let wasm = test_claim_wasm();
    let err = admit_v2(
        &worker(),
        &wasm,
        None,
        &[2u8, 0u8],
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(err.stage, 4);
    assert_eq!(err.code, Some(AbiRefusalCode::GrantsExceedLane));
    assert!(err.reason.contains("channel 9"), "{}", err.reason);
}

// -- under-claim: the hard-accountable cap traps ATTRIBUTABLY at run time (ABI §9.1) -----------------

#[test]
fn under_claim_traps_attributably_at_the_hard_cap() {
    let wasm = test_claim_wasm();
    let w = worker();
    let cfg = vec![3u8, 0u8]; // mode 3: claims 512 B hard host, stages 4096 B
    let admission = admit_v2(
        &w,
        &wasm,
        None,
        &cfg,
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        None,
        None,
    )
    .expect("the under-claimer's CLAIM is well-formed and within bounds — it admits");
    assert_eq!(admission.claim.hard_accountable.host, 512);

    let mut run_cfg = V2RunConfig::new(identity(&wasm, 2), [0x62; 32], cfg, Vec::new());
    run_cfg.claim_bytes = admission.claim_bytes;
    run_cfg.hard_accountable_host_bytes = admission.claim.hard_accountable.host;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&w, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    match run.wait().expect("join") {
        RunEnd::Trapped(trap) => {
            assert_eq!(trap.code, TrapCode::BudgetMemory);
            assert_eq!(trap.import, "stage_state");
            assert!(
                trap.detail.contains("attributable") && trap.detail.contains("512"),
                "the breach names the module's own claim: {}",
                trap.detail
            );
        }
        other => panic!("expected the attributable cap trap, got {other:?}"),
    }
    // The terminal fact is journaled as a trap (kind 1) naming the import.
    let entries = &sink.lock().expect("sink").entries;
    assert!(entries
        .iter()
        .any(|e| matches!(e, daemon_vhc_host::v2::SinkEntry::Terminal { kind: 1, .. })));
}

// -- D3 device-min admission pre-screen, admission side: the additive `device_min` pre-screen (funnel stage 3) ------------
//
// The three-proviso envelope fixture (the device-min envelope fixture) is green, so
// device-min admission pre-screen shipped interim-supported and stage 3 consumes the section — parsed from the RAW frozen
// bytes, threaded here exactly as the worker's `resolve_run → assess` path does. An envelope may
// only TIGHTEN the lane floor (§9.5).

/// Below the envelope's minimums: refused at stage 3, BEFORE any module work (fetch/compile).
#[test]
fn device_below_envelope_minimums_refuses_at_stage_3() {
    let min = daemon_vhc_proto::DeviceMinimums {
        ram_bytes: Some(64 << 30), // device() has 16 GiB
        ..daemon_vhc_proto::DeviceMinimums::default()
    };
    // Garbage wasm proves stage order: stage 3 must refuse before stage 4 ever sees the bytes.
    let err = admit_v2(
        &worker(),
        b"not wasm",
        None,
        &[],
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        Some(&min),
        None,
    )
    .unwrap_err();
    assert_eq!(err.stage, 3);
    assert!(err.reason.contains("envelope"), "{}", err.reason);
}

/// A GPU-required envelope on a GPU-less device: stage-3 refusal with the named cause.
#[test]
fn gpu_required_by_envelope_refuses_on_gpu_less_device() {
    let min = daemon_vhc_proto::DeviceMinimums {
        gpu: Some(2),
        ..daemon_vhc_proto::DeviceMinimums::default()
    };
    let err = admit_v2(
        &worker(),
        b"not wasm",
        None,
        &[],
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        Some(&min),
        None,
    )
    .unwrap_err();
    assert_eq!(err.stage, 3);
    assert!(err.reason.contains("GPU"), "{}", err.reason);
}

/// Minimums met: the funnel proceeds through assessment and admits the real v2 module —
/// the positive half of the device-min admission pre-screen admission gate.
#[test]
fn device_meeting_envelope_minimums_admits_the_module() {
    let wasm = test_claim_wasm();
    let min = daemon_vhc_proto::DeviceMinimums {
        gpu: Some(1),
        ram_bytes: Some(8 << 30),
        disk_bytes: Some(1 << 30),
        ..daemon_vhc_proto::DeviceMinimums::default()
    };
    let admission = admit_v2(
        &worker(),
        &wasm,
        Some(blake3::hash(&wasm).as_bytes()),
        &[0u8, 0u8],
        &[],
        &lane(),
        &device(),
        &owner_uncapped(),
        Some(&min),
        None,
    )
    .expect("minima met: the pre-screen passes and assessment admits");
    assert!(admission.claim.host_total() > 0);
}
