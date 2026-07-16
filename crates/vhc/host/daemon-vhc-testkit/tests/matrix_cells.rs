// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The D2 mixed-fleet matrix cells for {wasm coordinator × v1/v2 workers} (decisions D3;
// refactor §8/D2 acceptance):
//
// - **cell 8** (v2 worker × wasm coordinator × envelope v2) — the SUPPORTED target end-state,
//   pinned POSITIVE: a real whole run (production coordinator_quorum.wasm + production
//   tiny_llama_v2.wasm workers, both under the real major-2 driver), journaled, §8.7
//   replay-verified, cross-worker det-digest agreement.
// - **cell 3** (v1 worker × wasm coordinator × envelope v1) — REFUSED, typed: envelope v1 has no
//   coordinator role entry (no module-hash pin) and no Authority/identities section — a wasm
//   coordinator is unconfigurable from it.
// - **cell 7** (v2 worker × wasm coordinator × envelope v1) — REFUSED, typed: same
//   coordinator-side refusal; the worker's ABI axis cannot rescue an unconfigurable coordinator.
// - **cell 4** (v1 worker × wasm coordinator × envelope v2) — REFUSED, typed: the v1 five-phase
//   driver's config source is the v1 envelope's [data]/[phases], which envelope v2 does not carry;
//   the v1 opener refuses genesis bytes and the schema sniff routes v2 away from the v1 path.
//
// The cell-6 native-coordinator adapter (v2 × native × v2) is NOT retired here — D1 is
// building/threading it; retirement happens after D1's merge.
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern).
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_vhc_abi::CandidateDriver;
use daemon_vhc_host::v2::RunEnd;
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::{peek_schema, peer_id, Hash, SigningKey, GENESIS_SCHEMA_MAJOR};
use daemon_vhc_testkit::{
    cell8_genesis, cell8_whole_run, configure_wasm_coordinator, refuse_unconfigurable_envelope,
    Cell8Spec, WasmCoordError,
};

// -- guest build (the established testkit pattern) -------------------------------------------------

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

fn guest_wasm(name: &str) -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// A frozen schema-major-1 envelope's raw bytes (the barrier harness's v1 shape, signed).
fn frozen_v1_envelope_bytes() -> Vec<u8> {
    let envelope = daemon_vhc_testkit::barrier::barrier_envelope("matrix-v1", 2, 4, 2, 4);
    daemon_vhc_proto::to_canonical_vec(&envelope).expect("envelope cbor")
}

// -- cell 8: the SUPPORTED target end-state (positive whole-run gate) ------------------------------

/// Cell 8 single-worker: one production tiny-llama-v2 worker trains 2 barrier rounds under the
/// production wasm coordinator, configured from a genesis envelope v2 — journaled, replay-verified.
#[test]
fn cell8_single_worker_whole_run_is_green() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama_v2");
    let spec = Cell8Spec::new("cell8-1w", 1, 2);
    let report = cell8_whole_run(&coordinator, &worker, &spec).expect("whole run completes");

    assert_eq!(report.rounds_done, 2);
    assert_eq!(report.coordinator_records, 2, "one record per round");
    assert!(
        matches!(report.coordinator_end, RunEnd::Outcome(0)),
        "coordinator clean outcome, got {:?}",
        report.coordinator_end
    );
    let w = &report.workers[0];
    assert!(matches!(w.end, RunEnd::Outcome(0)));
    assert!(w.replay_matched, "§8.7 replay reproduced every decision");
    assert!(report.is_green());
}

/// Cell 8 multi-worker: 2 wasm workers under the wasm coordinator — the full inversion (no native
/// consensus anywhere in the run) — with cross-worker det-lane digest agreement as the oracle.
#[test]
fn cell8_two_workers_agree_on_the_det_lane_digest() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama_v2");
    let spec = Cell8Spec::new("cell8-2w", 2, 2);
    let report = cell8_whole_run(&coordinator, &worker, &spec).expect("whole run completes");

    assert_eq!(report.rounds_done, 2);
    assert_eq!(report.workers.len(), 2);
    for (i, w) in report.workers.iter().enumerate() {
        assert!(
            matches!(w.end, RunEnd::Outcome(0)),
            "worker {i} clean outcome, got {:?}",
            w.end
        );
        assert!(w.replay_matched, "worker {i} §8.7 replay matched");
    }
    assert_eq!(
        report.workers[0].digest, report.workers[1].digest,
        "cross-worker det-lane digest agreement under the wasm coordinator"
    );
    assert!(report.is_green());
}

// -- cells 3/7: wasm coordinator × envelope v1 — REFUSED, typed ------------------------------------

/// Cell 3: a v1 worker module (the pinned pre-refactor tiny-llama fixture — `select_driver`
/// proves the axis) under a wasm coordinator and envelope v1. The coordinator-side refusal fires
/// regardless of the worker: envelope v1 cannot pin a coordinator module or name an Authority.
#[test]
fn cell3_v1_worker_wasm_coordinator_envelope_v1_refused_typed() {
    // The worker-module axis: the v1 fixture selects the v1 five-phase driver.
    let v1_worker = guest_wasm("tiny_llama");
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v1_worker).as_bytes();
    let sel = select_driver(&engine, &v1_worker, Some(&hash)).expect("v1 selection");
    assert_eq!(sel.driver, CandidateDriver::V1, "the worker axis is v1");

    // The coordinator axis: envelope v1 cannot configure a wasm coordinator — typed refusal.
    let bytes = frozen_v1_envelope_bytes();
    assert_eq!(peek_schema(&bytes), Some(1));
    let err = refuse_unconfigurable_envelope(&bytes).unwrap_err();
    assert_eq!(err, WasmCoordError::EnvelopeCannotConfigure(1));
    // The refusal is a typed admission outcome that names the fix (a genesis envelope), never a
    // trap or a silent misparse.
    assert!(err.to_string().contains("genesis envelope v2"));
}

/// Cell 7: a v2 worker module under a wasm coordinator and envelope v1 — the same typed
/// coordinator-side refusal; a major-2 worker cannot rescue an unconfigurable coordinator.
#[test]
fn cell7_v2_worker_wasm_coordinator_envelope_v1_refused_typed() {
    let v2_worker = guest_wasm("tiny_llama_v2");
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v2_worker).as_bytes();
    let sel = select_driver(&engine, &v2_worker, Some(&hash)).expect("v2 selection");
    assert_eq!(sel.driver, CandidateDriver::V2, "the worker axis is v2");

    let bytes = frozen_v1_envelope_bytes();
    let err = refuse_unconfigurable_envelope(&bytes).unwrap_err();
    assert_eq!(err, WasmCoordError::EnvelopeCannotConfigure(1));
}

// -- cell 4: v1 worker × wasm coordinator × envelope v2 — REFUSED, typed ---------------------------

/// Cell 4: the v1 five-phase driver's config source is the v1 envelope's `[data]`/`[phases]`,
/// which the genesis schema does not carry (they became opaque module config at D0). The v1
/// opener refuses genesis bytes with a typed error, and the schema sniff routes v2 away from the
/// v1 path — never a silent misparse.
#[test]
fn cell4_v1_worker_wasm_coordinator_envelope_v2_refused_typed() {
    // The worker-module axis: v1.
    let v1_worker = guest_wasm("tiny_llama");
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v1_worker).as_bytes();
    let sel = select_driver(&engine, &v1_worker, Some(&hash)).expect("v1 selection");
    assert_eq!(sel.driver, CandidateDriver::V1);

    // A well-formed genesis envelope v2 (the same authoring the cell-8 positive uses).
    let coordinator = guest_wasm("coordinator_quorum");
    let coord_hash = Hash(*blake3::hash(&coordinator).as_bytes());
    let author = SigningKey::from_bytes(&[0x11u8; 32]);
    let genesis = cell8_genesis(
        "cell4-run",
        coord_hash,
        Hash(hash),
        peer_id(&author),
        1,
        2,
        2,
    );
    let frozen = genesis.freeze(&author).expect("genesis freeze");

    // The schema sniff routes the run AWAY from the v1 path (the dual-driver worker's gate).
    assert_eq!(peek_schema(frozen.bytes()), Some(GENESIS_SCHEMA_MAJOR));

    // And the v1 envelope opener REFUSES the genesis bytes typed — the v1 driver's config source
    // cannot exist under envelope v2 (cells 2/4, decisions D3).
    let sig = *frozen.signature();
    let signer = *frozen.signer();
    let err = daemon_vhc_proto::FrozenEnvelope::open(frozen.bytes().to_vec(), sig, signer)
        .expect_err("the v1 opener must refuse a genesis envelope");
    let msg = err.to_string();
    assert!(!msg.is_empty(), "typed SwarmProtoError refusal, got: {msg}");

    // The wasm coordinator, by contrast, configures fine from the same envelope — the refusal is
    // strictly the v1 worker's, which is what makes this cell 4 and not cell 3/7.
    configure_wasm_coordinator(&frozen).expect("coordinator side is configurable under v2");
}
