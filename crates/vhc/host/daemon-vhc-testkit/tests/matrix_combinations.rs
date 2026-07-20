// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The D2 mixed-fleet matrix combinations for {coordinator × v1/v2 workers} (decisions D3;
// refactor §8/D2 acceptance):
//
// - **end-state** (v2 worker × coordinator × envelope v2) — the SUPPORTED target end-state,
//   pinned POSITIVE: a real whole run (production coordinator_quorum.wasm + production
//   tiny_llama.wasm compute@2 trainers, both under the real major-2 driver), journaled, §8.7
//   replay-verified, cross-worker det-digest agreement.
// - **v1-worker/envelope-v1 refusal** (v1 worker × coordinator × envelope v1) — REFUSED, typed: envelope v1 has no
//   coordinator role entry (no module-hash pin) and no Authority/identities section — a wasm
//   coordinator is unconfigurable from it.
// - **v2-worker/envelope-v1 refusal** (v2 worker × coordinator × envelope v1) — REFUSED, typed: same
//   coordinator-side refusal; the worker's ABI axis cannot rescue an unconfigurable coordinator.
// - **v1-worker/envelope-v2 refusal** (v1 worker × coordinator × envelope v2) — REFUSED, typed: the v1 five-phase
//   driver's config source is the v1 envelope's [data]/[phases], which envelope v2 does not carry;
//   the v1 opener refuses genesis bytes and the schema sniff routes v2 away from the v1 path.
//
// The retired-native-coordinator native-coordinator adapter (v2 × native × v2) WAS retired at D2 (decisions D3
// retired-native-coordinator): a genesis run's coordination is its coordinator module (end-state).
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern).
#![allow(clippy::disallowed_methods)]

use daemon_vhc_abi::{AbiRefusalCode, CandidateDriver};
use daemon_vhc_host::run::RunEnd;
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::{peek_schema, peer_id, Hash, SigningKey, GENESIS_SCHEMA_MAJOR};
use daemon_vhc_testkit::{
    configure_coordinator, genesis_envelope, genesis_whole_run, refuse_unconfigurable_envelope,
    CoordError, GenesisRunSpec,
};

// -- guest build (the established testkit pattern) -------------------------------------------------

fn guest_wasm(name: &str) -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm(name)
}

/// Synthetic schema-major-1 envelope bytes: a canonical-CBOR map carrying `[run].schema = 1` —
/// the retired v1 form's outer shape. The refusal is decided by the outer schema-major read
/// alone (`peek_schema`), so the pin's input needs no retired v1 payload machinery.
fn synthetic_v1_envelope_bytes() -> Vec<u8> {
    use ciborium::value::Value;
    let run = Value::Map(vec![
        (Value::Text("schema".into()), Value::from(1u32)),
        (
            Value::Text("run_id".into()),
            Value::Text("matrix-v1".into()),
        ),
    ]);
    let envelope = Value::Map(vec![(Value::Text("run".into()), run)]);
    daemon_vhc_proto::to_canonical_vec(&envelope).expect("envelope cbor")
}

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
/// "v1 worker module" input the two v1-worker cells refuse typed at the §1.3 front door —
/// hand-built here (the `wasm-encoder` shape of `daemon-vhc-host/tests/driver_selection.rs`) so no
/// vendored/recorded pre-refactor bytes and no soon-to-retire v1 guest crate is load-bearing for
/// the refusal proof.
fn synthetic_v1_worker_module() -> Vec<u8> {
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

// -- end-state: the SUPPORTED target end-state (positive whole-run gate) ------------------------------

/// End-state single-worker: one production compute@2 trainer trains 2 barrier rounds under the
/// production coordinator, configured from a genesis envelope v2 — journaled, replay-verified.
#[test]
fn genesis_single_worker_whole_run_is_green() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama");
    let spec = GenesisRunSpec::new("genesis_run-1w", 1, 2);
    let report = genesis_whole_run(&coordinator, &worker, &spec).expect("whole run completes");

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

/// End-state multi-worker: 2 wasm workers under the coordinator — the full inversion (no native
/// consensus anywhere in the run) — with cross-worker det-lane digest agreement as the oracle.
#[test]
fn genesis_two_workers_agree_on_the_det_lane_digest() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama");
    let spec = GenesisRunSpec::new("genesis_run-2w", 2, 2);
    let report = genesis_whole_run(&coordinator, &worker, &spec).expect("whole run completes");

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
        "cross-worker det-lane digest agreement under the coordinator"
    );
    assert!(report.is_green());
}

// -- the envelope-v1 refusal combinations: coordinator × envelope v1 — REFUSED, typed ------------------------------------

/// v1-worker/envelope-v1 refusal: a v1 worker module under a coordinator and envelope v1. **Post-sunset the cell
/// is doubly refused** (decisions D5): the worker axis itself now meets the typed
/// `AbiUnsupportedMajor` at the §1.3 front door (the v1 driver retired — pre-sunset this
/// asserted `CandidateDriver::V1` selection), and the coordinator-side refusal fires regardless:
/// envelope v1 cannot pin a coordinator module or name an Authority.
#[test]
fn v1_worker_envelope_v1_refused_typed() {
    // The worker-module axis: a synthetic ABI-major-1 module is refused typed at driver selection
    // (the sunset) — the offending input is hand-assembled in-test, not a vendored/recorded fixture.
    let v1_worker = synthetic_v1_worker_module();
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v1_worker).as_bytes();
    let refusal = select_driver(&engine, &v1_worker, Some(&hash))
        .expect_err("the v1 worker axis is refused post-sunset");
    assert_eq!(refusal.code, AbiRefusalCode::AbiUnsupportedMajor);

    // The coordinator axis: envelope v1 cannot configure a coordinator — typed refusal.
    let bytes = synthetic_v1_envelope_bytes();
    assert_eq!(peek_schema(&bytes), Some(1));
    let err = refuse_unconfigurable_envelope(&bytes).unwrap_err();
    assert_eq!(err, CoordError::EnvelopeCannotConfigure(1));
    // The refusal is a typed admission outcome that names the fix (a genesis envelope), never a
    // trap or a silent misparse.
    assert!(err.to_string().contains("genesis envelope v2"));
}

/// v2-worker/envelope-v1 refusal: a v2 worker module under a coordinator and envelope v1 — the same typed
/// coordinator-side refusal; a major-2 worker cannot rescue an unconfigurable coordinator.
#[test]
fn v2_worker_envelope_v1_refused_typed() {
    let v2_worker = guest_wasm("tiny_llama");
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v2_worker).as_bytes();
    let sel = select_driver(&engine, &v2_worker, Some(&hash)).expect("v2 selection");
    assert_eq!(sel.driver, CandidateDriver::V2, "the worker axis is v2");

    let bytes = synthetic_v1_envelope_bytes();
    let err = refuse_unconfigurable_envelope(&bytes).unwrap_err();
    assert_eq!(err, CoordError::EnvelopeCannotConfigure(1));
}

// -- v1-worker/envelope-v2 refusal: v1 worker × coordinator × envelope v2 — REFUSED, typed ---------------------------

/// The remaining v1-worker cell: a v1 worker module under a coordinator and envelope v2.
/// The worker axis is refused typed at driver selection (`AbiUnsupportedMajor` — no v1 driver
/// exists), while the same genesis envelope configures the coordinator fine: the refusal is
/// strictly the worker's. (The retired v1 envelope OPENER this cell also used to exercise is
/// gone with the v1 envelope machinery — schema routing is the outer schema-major read, pinned
/// by the envelope-v1 cells above.)
#[test]
fn v1_worker_envelope_v2_refused_typed() {
    // The worker-module axis: a synthetic ABI-major-1 module (hand-assembled in-test, never a
    // vendored/recorded fixture) is refused typed at driver selection.
    let v1_worker = synthetic_v1_worker_module();
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v1_worker).as_bytes();
    let refusal =
        select_driver(&engine, &v1_worker, Some(&hash)).expect_err("the v1 worker axis is refused");
    assert_eq!(refusal.code, AbiRefusalCode::AbiUnsupportedMajor);

    // A well-formed genesis envelope v2 (the same authoring the whole-run positive uses).
    let coordinator = guest_wasm("coordinator_quorum");
    let coord_hash = Hash(*blake3::hash(&coordinator).as_bytes());
    let author = SigningKey::from_bytes(&[0x11u8; 32]);
    let genesis = genesis_envelope(
        "envelope-v2-refusal-run",
        coord_hash,
        Hash(hash),
        peer_id(&author),
        1,
        2,
        2,
    );
    let frozen = genesis.freeze(&author).expect("genesis freeze");

    // The schema read recognizes the genesis form...
    assert_eq!(peek_schema(frozen.bytes()), Some(GENESIS_SCHEMA_MAJOR));
    // ...and the coordinator configures fine from the same envelope — the refusal is
    // strictly the v1 worker's, which is what makes this the v1-worker cell and not an
    // envelope-v1 cell.
    configure_coordinator(&frozen).expect("coordinator side is configurable under v2");
}
