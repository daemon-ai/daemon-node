// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// E1 checkpoint bridge over the REAL guest (refactor §9 Phase E entry item; architecture §5.3):
// a `WasmBackend` running the actual `tiny_llama.wasm` saves a **typed** checkpoint — the module
// section is the safetensors serialization of its param masters in registration order (the
// `daemon-vhc-safetensors` wiring), the opaque `checkpoint_save` bytes ride as the authoritative
// `worker-local` section, and the whole thing is content-addressed on the payload plane. A fresh
// backend restores from it bit-exactly (same digest on the identical next ingest), and the typed
// section re-imports as the exact state dict that was exported.
//
// Tier-1 (runs in the default `daemon-swarm-e2e` lane: no iroh, no live, CPU-deterministic).
//
// Test harness that shells `cargo build` for the guests + reads the built `.wasm` from disk (the
// wasm_profiles.rs pattern), so the fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Once};

use ciborium::into_writer;

use daemon_vhc_host::runtime::EngineConfig;
use daemon_vhc_net::{FsPayloadStore, PayloadStore};
use daemon_vhc_proto::blake3_hash;
use daemon_vhc_sdk::models::TinyLlamaCfg;
use daemon_vhc_sdk_consensus::checkpoint::{SectionClass, SectionKind};
use daemon_vhc_session::backend::{BatchRef, StateDict, StepCtx, TrainerBackend};
use daemon_vhc_session::checkpoint::{
    load_typed_checkpoint, save_typed_checkpoint, typed_section_key, CheckpointCapture,
    CheckpointIdent, MODULE_SCHEMA_OPAQUE, MODULE_SCHEMA_SAFETENSORS,
};
use daemon_vhc_session::seam::RunId;
use daemon_vhc_session::{WasmBackend, WasmBackendConfig};

// -- guest module loading (the wasm_profiles.rs pattern) ------------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/vhc/guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
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

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
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
    });
}

fn tiny_llama_wasm() -> Vec<u8> {
    let path = guest_dir().join("tiny_llama.wasm");
    if !path.exists() {
        ensure_built();
    }
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// -- fixture --------------------------------------------------------------------------------------

const SEQ: u32 = 8;
const VOCAB: u32 = 64;

fn cfg_cbor() -> Vec<u8> {
    let cfg = TinyLlamaCfg {
        n_layers: 1,
        seq_len: SEQ + 1,
        vocab: VOCAB,
        ..TinyLlamaCfg::default()
    };
    let mut b = Vec::new();
    into_writer(&cfg, &mut b).expect("cbor");
    b
}

fn make_backend(config: &[u8]) -> WasmBackend {
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: tiny_llama_wasm(),
        engine: EngineConfig::default(),
    })
    .expect("construct WasmBackend");
    b.build(config).expect("da_build");
    b
}

fn tokens(salt: u64) -> Vec<u32> {
    (0..SEQ * 2)
        .map(|i| {
            ((u64::from(i).wrapping_mul(2_654_435_761).wrapping_add(salt)) % u64::from(VOCAB))
                as u32
        })
        .collect()
}

fn train_one_step(b: &mut WasmBackend, salt: u64) {
    let toks = tokens(salt);
    b.train_step(
        &BatchRef {
            seq_len: SEQ,
            tokens: toks,
        },
        StepCtx {
            inner_step: 0,
            mb_index: 0,
            mb_count: 1,
            step_seqs: 2,
        },
    )
    .expect("da_step");
    b.inner_update(0).expect("da_inner_update");
}

fn temp_store() -> Arc<FsPayloadStore> {
    let root = std::env::temp_dir().join(format!(
        "dvhc-typed-ckpt-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    Arc::new(FsPayloadStore::open(&root, 64).expect("fs store"))
}

fn staged(peer: u8, bytes: Vec<u8>) -> daemon_vhc_session::backend::StagedPayload {
    daemon_vhc_session::backend::StagedPayload {
        peer: daemon_vhc_proto::PeerId([peer; 32]),
        hash: blake3_hash(&bytes),
        bytes,
    }
}

// -- the test ---------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wasm_typed_checkpoint_bridges_safetensors_and_restores_bit_exactly() {
    let config = cfg_cbor();
    let run = RunId::new("typed-ckpt-wasm");
    let store = temp_store();
    let ident = CheckpointIdent {
        run_id: blake3_hash(b"typed-ckpt-wasm-genesis"),
        epoch: 0,
        module: blake3_hash(&tiny_llama_wasm()),
    };

    // Train a step so the state is non-initial, then round-0 ingest for a digest.
    let mut a = make_backend(&config);
    train_one_step(&mut a, 0x9E37_79B9);
    let upd = a.make_update(0).expect("da_make_update");
    let set = [staged(1, upd)];
    let d0 = a.ingest(0, &set).expect("da_ingest");

    // Save the typed checkpoint. The WasmBackend exports its param masters, so the module section
    // MUST be the safetensors typed serialization + the opaque worker-local restore section.
    let saved = save_typed_checkpoint(
        &store,
        &run,
        &ident,
        &a,
        CheckpointCapture {
            round: 0,
            digest: d0,
            data_cursor: 16,
            journal_position: 5,
        },
    )
    .await
    .expect("save typed checkpoint");
    let m = &saved.manifest;
    assert_eq!(m.run_id, ident.run_id);
    assert_eq!(m.module, ident.module);
    let module_sec = m.section(SectionKind::Module).expect("module section");
    assert_eq!(module_sec.schema, MODULE_SCHEMA_SAFETENSORS);
    assert_eq!(module_sec.class, SectionClass::RoleLocal);
    assert_eq!(
        m.section(SectionKind::WorkerLocal).expect("opaque").schema,
        MODULE_SCHEMA_OPAQUE
    );
    assert_eq!(
        m.section(SectionKind::Consensus).expect("consensus").class,
        SectionClass::ConsensusCanonical
    );

    // The typed module section re-imports as EXACTLY the state dict the live instance exported —
    // names, shapes, registration order, fp32 bits (the safetensors bridge, bit-exact).
    let st_bytes = store
        .get(
            &typed_section_key(&run, 0, SectionKind::Module),
            &module_sec.hash,
        )
        .await
        .expect("fetch module section");
    assert_eq!(blake3_hash(&st_bytes), module_sec.hash);
    let imported = StateDict::from_safetensors(&st_bytes).expect("valid safetensors");
    let exported = a
        .export_state_dict()
        .expect("export")
        .expect("wasm backend exports");
    assert_eq!(
        imported, exported,
        "safetensors round-trip reproduces the exported state dict bit-for-bit"
    );
    assert!(
        !imported.tensors.is_empty(),
        "the guest registered parameters"
    );

    // Restore into a FRESH backend (fresh instance, same config): the authoritative worker-local
    // opaque bytes restore bit-exactly — the identical next ingest reaches the identical digest.
    let mut b = make_backend(&config);
    train_one_step(&mut b, 0xDEAD_BEEF); // diverge b first, so the restore is doing real work
    load_typed_checkpoint(&store, &run, &mut b, &saved.pointer)
        .await
        .expect("load typed checkpoint");

    let next_upd_a = a.make_update(1).expect("update a");
    let next_upd_b = b.make_update(1).expect("update b");
    assert_eq!(
        next_upd_a, next_upd_b,
        "restored state reproduces the source's next update bit-for-bit"
    );
    let next_set = [staged(1, next_upd_a)];
    assert_eq!(
        a.ingest(1, &next_set).expect("ingest a"),
        b.ingest(1, &next_set).expect("ingest b"),
        "restored peer reaches the in-sync digest on the identical next ingest"
    );
}
