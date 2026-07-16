// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Capture the **v1 parity oracle** as a content-addressed recorded fixture (Phase E sunset
//! prerequisite; decisions D5 — the frozen parity pins survive the v1 driver's deletion as
//! recorded-oracle regressions).
//!
//! Reproduces, byte-for-byte, the two live v1 oracles the parity tests ran before the sunset:
//!
//! - the `v2_parity` oracle: `daemon_vhc_session::WasmBackend` (the v1 five-phase driver) over
//!   the current `tiny_llama` guest, CPU `EngineConfig::default()`, 2 rounds × 2 steps of
//!   all-zero tokens, single-peer self-ingest — recording the per-round sealed update bytes and
//!   the final post-ingest det-lane state digest;
//! - the `c3_parity` oracle: the same guest driven through `daemon_vhc_host::Instance` directly
//!   (burn-ndarray backend), varied tokens — recording matched init θ, per-round trained θ
//!   (post-inner-steps, pre-ingest), per-round committed payload bytes, and per-round
//!   post-ingest digests, plus the canonical param names/numels.
//!
//! The recording is backend-independent by the standing det-lane bit-identity invariant
//! (refactor §12.1; `burn_backend_parity.rs` pins every v1 op bit-exact across backends), so one
//! CPU-lane capture serves the cpu/burn-ndarray/wgpu/cuda parity tiers alike.
//!
//! Run from THIS directory on the pre-sunset tree, with the guests built (see ../README.md).

use std::path::{Path, PathBuf};

use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::{blake3_hash, digest_state, PeerId, Seed};
use daemon_vhc_sdk::models::TinyLlamaCfg;
use daemon_vhc_session::backend::{BatchRef, StagedPayload, StepCtx, TrainerBackend};
use daemon_vhc_session::{WasmBackend, WasmBackendConfig};

const ROUNDS: u64 = 2;
const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 2;
const SEQ_LEN: u32 = 9;
const VOCAB: u32 = 64;
const PEER: [u8; 32] = [7u8; 32];

fn fixture_root() -> PathBuf {
    // capture/ lives inside the fixture directory.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fixture root")
        .to_path_buf()
}

fn guests_release() -> PathBuf {
    // <checkout>/crates/vhc/guests/target/wasm32-unknown-unknown/release, resolved relative to
    // this crate (…/host/daemon-vhc-host/tests/fixtures/v1-parity-oracle/capture).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../../guests/target/wasm32-unknown-unknown/release")
        .canonicalize()
        .expect("guests release dir (run `cargo run -p xtask -- build-guests` first)")
}

fn model_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: SEQ_LEN,
        ..TinyLlamaCfg::default()
    }
}

fn model_cfg_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(&model_cfg(), &mut b).expect("cfg cbor");
    b
}

fn zero_tokens() -> Vec<u32> {
    vec![0u32; (MICRO_BATCH * SEQ_LEN) as usize]
}

/// The c3 lane's deterministic varied tokens for `(round, step)`.
fn tokens_for(round: u64, step: u32) -> Vec<u32> {
    let n = (MICRO_BATCH * SEQ_LEN) as u64;
    (0..n)
        .map(|i| {
            let x = i + 1_000 * u64::from(step) + 100_000 * round + 1;
            (x.wrapping_mul(2_654_435_761) % u64::from(VOCAB)) as u32
        })
        .collect()
}

/// The digest exactly as `WasmBackend::digest_of` computes it (round-seeded, full sampling).
fn digest_of_state(state: &[u8], round: u64) -> [u8; 16] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&round.to_le_bytes());
    let d = digest_state(&Seed(seed), 64, u32::MAX, state);
    *d.as_bytes()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn write_file(root: &Path, rel: &str, bytes: &[u8]) -> serde_json::Value {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(&path, bytes).expect("write fixture file");
    serde_json::json!({
        "file": rel,
        "blake3": hex(blake3_hash(bytes).as_bytes()),
        "bytes": bytes.len(),
    })
}

/// Oracle A — the `v2_parity` shape: WasmBackend on CPU, zero tokens, self-ingest per round.
fn capture_v2_parity_oracle(wasm: &[u8]) -> (Vec<Vec<u8>>, [u8; 16]) {
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: wasm.to_vec(),
        engine: EngineConfig::default(),
    })
    .expect("v1 backend");
    b.build(&model_cfg_bytes()).expect("build");
    let peer = PeerId(PEER);
    let mut updates = Vec::new();
    let mut last = None;
    for round in 0..ROUNDS {
        for h in 0..STEPS_PER_ROUND {
            b.train_step(
                &BatchRef {
                    tokens: zero_tokens(),
                    seq_len: SEQ_LEN,
                },
                StepCtx {
                    inner_step: h,
                    mb_index: 0,
                    mb_count: 1,
                    step_seqs: MICRO_BATCH,
                },
            )
            .expect("step");
            b.inner_update(h).expect("inner");
        }
        let bytes = b.make_update(round).expect("make_update");
        let digest = b
            .ingest(
                round,
                &[StagedPayload {
                    peer,
                    hash: blake3_hash(&bytes),
                    bytes: bytes.clone(),
                }],
            )
            .expect("ingest");
        updates.push(bytes);
        last = Some(digest);
    }
    (updates, last.expect("rounds ran").0)
}

struct C3Oracle {
    names: Vec<String>,
    numels: Vec<usize>,
    init: Vec<Vec<f32>>,
    trained: Vec<Vec<Vec<f32>>>,
    payloads: Vec<Vec<u8>>,
    digests: Vec<[u8; 16]>,
}

/// Oracle B — the `c3_parity` shape: the Instance API on burn-ndarray, varied tokens.
fn capture_c3_oracle(wasm: &[u8]) -> C3Oracle {
    let engine = EngineConfig {
        backend: BackendKind::BurnNdarray,
        ..EngineConfig::default()
    };
    let worker = Worker::new(engine).expect("v1 engine");
    let module = worker.load_module(wasm).expect("v1 module");
    let mut inst = worker.instantiate(&module).expect("v1 instance");
    inst.build(&model_cfg_bytes()).expect("da_build");

    let params = inst.params();
    let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
    let read_state = |inst: &daemon_vhc_host::Instance| -> Vec<Vec<f32>> {
        names
            .iter()
            .map(|n| inst.param_master(n).expect("param master"))
            .collect()
    };
    let init = read_state(&inst);
    let numels: Vec<usize> = init.iter().map(Vec::len).collect();

    let mut trained = Vec::new();
    let mut payloads = Vec::new();
    let mut digests = Vec::new();
    for round in 0..ROUNDS {
        for h in 0..STEPS_PER_ROUND {
            let bh = inst.register_batch(tokens_for(round, h), MICRO_BATCH, SEQ_LEN);
            inst.step(bh, h, 0, 1, MICRO_BATCH).expect("da_step");
            inst.inner_update(h).expect("da_inner_update");
        }
        trained.push(read_state(&inst));
        let container = inst.make_update(round).expect("da_make_update");
        let payload = inst.update_bytes(container).expect("update bytes");
        inst.ingest_payloads(round, std::slice::from_ref(&payload))
            .expect("ingest");
        payloads.push(payload);
        digests.push(digest_of_state(&inst.canonical_state_bytes(), round));
    }
    C3Oracle {
        names,
        numels,
        init,
        trained,
        payloads,
        digests,
    }
}

fn flat_le(params: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(params.iter().map(Vec::len).sum::<usize>() * 4);
    for p in params {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn main() {
    let root = fixture_root();
    let wasm_path = guests_release().join("tiny_llama.wasm");
    let wasm = std::fs::read(&wasm_path)
        .unwrap_or_else(|e| panic!("read {} — build the guests first: {e}", wasm_path.display()));
    let commit = std::env::var("CAPTURE_COMMIT").unwrap_or_else(|_| "UNSET".into());

    // -- oracle A (v2_parity) ---------------------------------------------------------------
    let (updates, final_digest) = capture_v2_parity_oracle(&wasm);
    let model_cfg_entry = write_file(&root, "model-cfg.v1.cbor", &model_cfg_bytes());
    let update_entries: Vec<serde_json::Value> = updates
        .iter()
        .enumerate()
        .map(|(r, u)| write_file(&root, &format!("updates/v2p-round-{r}.bin"), u))
        .collect();

    // -- oracle B (c3_parity) ---------------------------------------------------------------
    let c3 = capture_c3_oracle(&wasm);
    let init_entry = write_file(&root, "c3/init.f32le.bin", &flat_le(&c3.init));
    let trained_entries: Vec<serde_json::Value> = c3
        .trained
        .iter()
        .enumerate()
        .map(|(r, t)| {
            write_file(
                &root,
                &format!("c3/trained-round-{r}.f32le.bin"),
                &flat_le(t),
            )
        })
        .collect();
    let payload_entries: Vec<serde_json::Value> = c3
        .payloads
        .iter()
        .enumerate()
        .map(|(r, p)| write_file(&root, &format!("c3/payload-round-{r}.bin"), p))
        .collect();

    // The exact config literals the post-sunset tests reconstruct their configs from (recorded
    // from the live TinyLlamaCfg so nothing is transcribed by hand).
    let m = model_cfg();
    let expected = serde_json::json!({
        "captured_from": {
            "commit": commit,
            "tree": "vhc/e3-sunset (pre-sunset — the last tree with the live v1 driver)",
            "driver": "v1 five-phase (daemon-vhc-session WasmBackend / daemon-vhc-host Instance)",
            "capture": "tests/fixtures/v1-parity-oracle/capture (see README.md)",
        },
        "module": {
            "name": "tiny_llama.wasm",
            "blake3": hex(blake3_hash(&wasm).as_bytes()),
            "bytes": wasm.len(),
            "source": "guests/target build at capture (byte-identical across checkout paths \
                       via the guests workspace rustc shim; pinned in guests.blake3 at the \
                       capture commit)",
        },
        "backend_independence": "det-lane bit-identity (refactor §12.1) + the v1 op backend \
             parity suites (burn_backend_parity.rs) make this recording valid for every \
             backend tier (cpu / burn-ndarray / wgpu / cuda)",
        "v2_parity": {
            "engine": "cpu (EngineConfig::default())",
            "schedule": {
                "rounds": ROUNDS, "steps_per_round": STEPS_PER_ROUND,
                "micro_batch": MICRO_BATCH, "seq_len": SEQ_LEN,
                "peer": "[7; 32]", "tokens": "all-zero",
            },
            "model_cfg": model_cfg_entry,
            "updates": update_entries,
            "final_digest": hex(&final_digest),
        },
        "c3_parity": {
            "engine": "burn-ndarray",
            "schedule": {
                "rounds": ROUNDS, "steps_per_round": STEPS_PER_ROUND,
                "micro_batch": MICRO_BATCH, "seq_len": SEQ_LEN,
                "peer": "[7; 32]",
                "tokens": "token[i] of (round, step) = ((i + 1000*step + 100000*round + 1) \
                           * 2654435761) % 64",
            },
            "model_cfg": {
                "d_model": m.d_model, "n_layers": m.n_layers, "n_heads": m.n_heads,
                "head_dim": m.head_dim, "vocab": m.vocab, "seq_len": m.seq_len,
                "ffn_mult": m.ffn_mult, "rope_theta": m.rope_theta,
                "rmsnorm_eps": m.rmsnorm_eps, "lr": m.inner.lr, "beta1": m.inner.beta1,
                "beta2": m.inner.beta2, "adam_eps": m.inner.eps, "wd": m.inner.wd,
            },
            "profile_cfg": {
                "h": m.sparse_loco.h, "ef_decay": m.sparse_loco.ef_decay,
                "chunk": m.sparse_loco.chunk, "topk": m.sparse_loco.topk,
                "bits": m.sparse_loco.bits, "outer_alpha": m.sparse_loco.outer_alpha,
                "clip": m.sparse_loco.clip,
            },
            "param_names": c3.names,
            "param_numels": c3.numels,
            "init": init_entry,
            "trained": trained_entries,
            "payloads": payload_entries,
            "digests": c3.digests.iter().map(|d| hex(d)).collect::<Vec<_>>(),
        },
    });
    let json = serde_json::to_string_pretty(&expected).expect("expected json");
    std::fs::write(root.join("expected.json"), json + "\n").expect("write expected.json");
    println!(
        "captured v1 parity oracle into {} (module {} … final digest {})",
        root.display(),
        &hex(blake3_hash(&wasm).as_bytes())[..16],
        hex(&final_digest),
    );
}
