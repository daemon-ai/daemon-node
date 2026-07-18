// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The D2 mixed-fleet matrix cells for {wasm coordinator × v1/v2 workers} (decisions D3;
// refactor §8/D2 acceptance):
//
// - **cell 8** (v2 worker × wasm coordinator × envelope v2) — the SUPPORTED target end-state,
//   pinned POSITIVE: a real whole run (production coordinator_quorum.wasm + production
//   tiny_llama_c3.wasm compute@2 trainers, both under the real major-2 driver), journaled, §8.7
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
// The cell-6 native-coordinator adapter (v2 × native × v2) WAS retired at D2 (decisions D3
// cell 6): a genesis run's coordination is its wasm coordinator module (cell 8).
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern).
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_vhc_abi::{AbiRefusalCode, CandidateDriver};
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

// -- cell 8: the SUPPORTED target end-state (positive whole-run gate) ------------------------------

/// Cell 8 single-worker: one production compute@2 trainer trains 2 barrier rounds under the
/// production wasm coordinator, configured from a genesis envelope v2 — journaled, replay-verified.
#[test]
fn cell8_single_worker_whole_run_is_green() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama_c3");
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
    let worker = guest_wasm("tiny_llama_c3");
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

/// Cell 3: a v1 worker module under a wasm coordinator and envelope v1. **Post-sunset the cell
/// is doubly refused** (decisions D5): the worker axis itself now meets the typed
/// `AbiUnsupportedMajor` at the §1.3 front door (the v1 driver retired — pre-sunset this
/// asserted `CandidateDriver::V1` selection), and the coordinator-side refusal fires regardless:
/// envelope v1 cannot pin a coordinator module or name an Authority.
#[test]
fn cell3_v1_worker_wasm_coordinator_envelope_v1_refused_typed() {
    // The worker-module axis: a synthetic ABI-major-1 module is refused typed at driver selection
    // (the sunset) — the offending input is hand-assembled in-test, not a vendored/recorded fixture.
    let v1_worker = synthetic_v1_worker_module();
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v1_worker).as_bytes();
    let refusal = select_driver(&engine, &v1_worker, Some(&hash))
        .expect_err("the v1 worker axis is refused post-sunset");
    assert_eq!(refusal.code, AbiRefusalCode::AbiUnsupportedMajor);

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
    let v2_worker = guest_wasm("tiny_llama_c3");
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
    // The worker-module axis: v1 — refused typed at driver selection since the Phase-E sunset
    // (pre-sunset this asserted `CandidateDriver::V1` selection; the cell stays refused, now
    // doubly: unsupported worker major AND the v1 opener refusing genesis bytes below). The v1
    // worker module is synthetic (hand-assembled in-test), not a vendored/recorded fixture.
    let v1_worker = synthetic_v1_worker_module();
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let hash = *blake3::hash(&v1_worker).as_bytes();
    let refusal = select_driver(&engine, &v1_worker, Some(&hash))
        .expect_err("the v1 worker axis is refused post-sunset");
    assert_eq!(refusal.code, AbiRefusalCode::AbiUnsupportedMajor);

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
