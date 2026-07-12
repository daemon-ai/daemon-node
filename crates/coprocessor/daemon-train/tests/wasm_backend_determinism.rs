// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The MVP's core claim (ABI §7): two `WasmBackend` peers over the same module + config + batches +
// staged set are bit-identical every round (cross-PEER determinism), across each of the three comm
// profiles; a checkpoint save→load→continue — and a preemption-as-churn pause→resume — reproduce the
// uninterrupted digest bit-for-bit. Plus a documented sim↔host cross-check.
//
// The `.wasm` is located via `SWARM_TEST_GUEST_DIR` if set, else built on demand (exactly what
// `xtask build-guests` does), so `cargo test --workspace` never silently skips. The dev-shell
// `wasm32-unknown-unknown` rust-std is required. This is a dev/test harness (it shells `cargo build`
// for the guests), so the fs/process bans (which target the shipped node) are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_swarm_proto::{blake3_hash, PeerId};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StateDigest, StepCtx, TrainerBackend};
use daemon_train::{EngineConfig, WasmBackend, WasmBackendConfig};
use daemon_train_sdk::models::TinyLlamaCfg;
use serde::Serialize;

// -- guest module loading (mirrors tests/guest_lifecycle.rs) ------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
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

fn cbor<T: Serialize>(v: &T) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(v, &mut b).expect("cbor");
    b
}

/// A small, fast tiny-llama config (1 layer) wired to `profile`. Dimensions are multiples of 64, so
/// every profile's chunking (`sparse_loco.chunk = 64` / `demo.tile² = 64`) needs no padding.
fn tiny_cfg(profile: &str) -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        profile: profile.to_string(),
        ..TinyLlamaCfg::default()
    }
}

fn backend(config: &[u8]) -> WasmBackend {
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: tiny_llama_wasm(),
        engine: EngineConfig::default(),
    })
    .expect("construct WasmBackend");
    b.build(config).expect("da_build");
    b
}

const SEQ: u32 = 8;
const SEQS: u32 = 2; // sequences per micro-batch

/// Deterministic token ids (< vocab 64) keyed by `salt` (a per-peer / per-round seed).
fn tokens(salt: u64) -> Vec<u32> {
    (0..SEQ * SEQS)
        .map(|i| ((u64::from(i).wrapping_mul(2_654_435_761).wrapping_add(salt)) % 64) as u32)
        .collect()
}

fn batch(salt: u64) -> BatchRef {
    BatchRef {
        tokens: tokens(salt),
        seq_len: SEQ,
    }
}

fn ctx(inner_step: u32) -> StepCtx {
    StepCtx {
        inner_step,
        mb_index: 0,
        mb_count: 1,
        step_seqs: SEQS,
    }
}

/// Train `steps_per_round` inner steps over batches seeded by `salt`, then seal the round payload.
fn train_round(b: &mut WasmBackend, steps_per_round: u32, round: u64, salt: u64) -> Vec<u8> {
    for step in 0..steps_per_round {
        b.train_step(&batch(salt ^ u64::from(step)), ctx(step))
            .expect("train_step");
        b.inner_update(step).expect("inner_update");
    }
    b.make_update(round).expect("make_update")
}

fn staged(peer: u8, bytes: &[u8]) -> StagedPayload {
    StagedPayload {
        peer: PeerId([peer; 32]),
        hash: blake3_hash(bytes),
        bytes: bytes.to_vec(),
    }
}

// -- cross-PEER bit identity (the guarantee) ----------------------------------------------------

fn cross_peer_agrees(profile: &str, rounds: u64) -> Vec<StateDigest> {
    let config = cbor(&tiny_cfg(profile));
    let mut a = backend(&config);
    let mut b = backend(&config);
    let steps = a.steps_per_round().expect("steps_per_round");
    assert!(steps >= 1, "cadence must be positive");

    let mut transcript = Vec::new();
    for round in 0..rounds {
        // Two peers, same config + same batches (the MVP claim's premise): identical local math →
        // identical round payloads. The host `CpuBackend` now runs real reverse-mode autodiff
        // (HOST-9), so `make_update` is data-dependent; feeding both peers identical batches keeps
        // the contributions equal, and the cross-peer bit-identity asserted below is the frozen
        // guarantee that holds because both peers run the same deterministic fp32 arithmetic.
        let salt = 0x1234 ^ round;
        let pa = train_round(&mut a, steps, round, salt);
        let pb = train_round(&mut b, steps, round, salt);
        assert_eq!(
            pa, pb,
            "{profile} r{round}: same config + batches must yield the same contribution"
        );
        // Both ingest the identical committed set in identical record order → equal digest.
        let set = vec![staged(0x01, &pa), staged(0x02, &pb)];
        let da = a.ingest(round, &set).expect("ingest a");
        let db = b.ingest(round, &set).expect("ingest b");
        assert_eq!(
            da,
            db,
            "{profile} r{round}: cross-peer digest must be bit-identical (got {} vs {})",
            da.to_hex(),
            db.to_hex()
        );
        transcript.push(da);
    }
    // Non-degenerate: the round loop actually evolves the canonical state (not a vacuous constant).
    if transcript.len() >= 2 {
        assert!(
            transcript.windows(2).any(|w| w[0] != w[1]),
            "{profile}: the digest transcript must evolve across rounds"
        );
    }
    transcript
}

#[test]
fn cross_peer_bit_identity_sparse_loco() {
    let t = cross_peer_agrees("sparse_loco", 3);
    assert_eq!(t.len(), 3);
}

#[test]
fn cross_peer_bit_identity_diloco() {
    let t = cross_peer_agrees("diloco", 3);
    assert_eq!(t.len(), 3);
}

#[test]
fn cross_peer_bit_identity_demo() {
    let t = cross_peer_agrees("demo", 3);
    assert_eq!(t.len(), 3);
}

/// The whole transcript is reproducible run-to-run (determinism across process re-instantiation).
#[test]
fn transcript_is_reproducible() {
    let a = cross_peer_agrees("sparse_loco", 3);
    let b = cross_peer_agrees("sparse_loco", 3);
    assert_eq!(a, b, "the digest transcript must be reproducible");
}

// -- checkpoint continuity + preemption-as-churn ------------------------------------------------

/// A single peer's self-ingest transcript (one payload committed per round), driven identically so a
/// checkpoint/resume mid-stream can be compared against the uninterrupted reference.
fn self_run(
    b: &mut WasmBackend,
    steps: u32,
    rounds_start: u64,
    rounds_end: u64,
) -> Vec<StateDigest> {
    let mut out = Vec::new();
    for round in rounds_start..rounds_end {
        let payload = train_round(b, steps, round, 0x5EED ^ round);
        let digest = b
            .ingest(round, &[staged(0x01, &payload)])
            .expect("self ingest");
        out.push(digest);
    }
    out
}

#[test]
fn checkpoint_save_load_continue_matches_uninterrupted() {
    let config = cbor(&tiny_cfg("sparse_loco"));
    let steps = {
        let mut probe = backend(&config);
        probe.steps_per_round().unwrap()
    };

    // Reference: an uninterrupted 4-round run.
    let mut reference = backend(&config);
    let ref_all = self_run(&mut reference, steps, 0, 4);

    // Interrupted: run rounds 0..2, checkpoint, then continue rounds 2..4 in a FRESH backend that
    // built from the same config and loaded the checkpoint.
    let mut first = backend(&config);
    let _ = self_run(&mut first, steps, 0, 2);
    let checkpoint = first.checkpoint_save().expect("checkpoint_save");

    let mut resumed = backend(&config);
    resumed
        .checkpoint_load(&checkpoint)
        .expect("checkpoint_load");
    let tail = self_run(&mut resumed, steps, 2, 4);

    assert_eq!(
        tail,
        ref_all[2..].to_vec(),
        "save→load→continue must reproduce the uninterrupted digest transcript"
    );
}

#[test]
fn preemption_as_churn_is_digest_neutral() {
    let config = cbor(&tiny_cfg("diloco"));
    let steps = {
        let mut probe = backend(&config);
        probe.steps_per_round().unwrap()
    };

    let mut reference = backend(&config);
    let ref_all = self_run(&mut reference, steps, 0, 4);

    // Churned: run 0..2, pause (checkpoint + drop the wasm instance, keep CPU masters), resume
    // (re-instantiate + da_build + restore), continue 2..4 — must match the uninterrupted run.
    let mut churned = backend(&config);
    let _ = self_run(&mut churned, steps, 0, 2);
    churned.pause().expect("pause");
    churned.resume().expect("resume");
    let tail = self_run(&mut churned, steps, 2, 4);

    assert_eq!(
        tail,
        ref_all[2..].to_vec(),
        "preemption-as-churn (pause→resume) must be digest-neutral"
    );
}

// -- sim ↔ host parity (documented cross-check) -------------------------------------------------

// The MVP guarantee is cross-PEER bit-identity (WasmBackend vs WasmBackend, one implementation),
// asserted above. Sim (`daemon-train-sdk` `sim`) vs host (`WasmBackend`/`CpuBackend`) equality is a
// nice-to-have that does NOT hold bit-for-bit, because the two seed their param init RNG differently
// (the host seeds via `runtime::fake_init` FNV/xorshift by name; the sim via `sim::init_values`
// splitmix/Box-Muller), so even the *initial* masters differ before a single step. Both now run a
// real reverse-mode tape over the shared det-core kernels (HOST-9), so both learn from data; only
// the det-lane ingest fold is a required numerics reference (ABI §7), and it is bit-identical.
//
// So sim-vs-host is exercised as *each backend is internally deterministic* (a re-run reproduces its
// own transcript), which is the property that matters for reproducibility; cross-lane bit-equality is
// intentionally not asserted.
#[test]
fn host_backend_is_self_consistent() {
    let a = cross_peer_agrees("demo", 2);
    let b = cross_peer_agrees("demo", 2);
    assert_eq!(a, b, "the host backend is internally deterministic");
    assert!(
        a.windows(2).all(|w| w[0] != w[1]) || a.len() < 2,
        "successive rounds evolve the state"
    );
}

// -- HOST-9: the host learns from data (loss decreases) -----------------------------------------

/// HOST-9 acceptance: with reverse-mode autodiff on the host path, a `WasmBackend` overfitting a
/// **fixed** synthetic batch drives the tiny-llama cross-entropy loss **down** over rounds (mirrors
/// E2's sim evidence, now on the real host `CpuBackend`). Before HOST-9 the host `backward` was a
/// no-op and the loss was flat; this test would have failed then.
fn loss_decreases_for(profile: &str) -> (f32, f32, Vec<f32>) {
    let config = cbor(&tiny_cfg(profile));
    let mut b = backend(&config);
    let steps = b.steps_per_round().expect("steps_per_round");
    // A single fixed batch (same tokens every step + round): the loss must fall as the host learns.
    let fixed = batch(0xF00D);
    let mut per_round = Vec::new();
    let mut first = f32::NAN;
    for round in 0..8 {
        let mut last = f32::NAN;
        for step in 0..steps {
            let stats = b.train_step(&fixed, ctx(step)).expect("train_step");
            if first.is_nan() {
                first = stats.loss;
            }
            last = stats.loss;
            b.inner_update(step).expect("inner_update");
        }
        // Keep the round loop honest (make_update + self-ingest exactly like a real round).
        let payload = b.make_update(round).expect("make_update");
        b.ingest(round, &[staged(0x01, &payload)]).expect("ingest");
        per_round.push(last);
    }
    (first, *per_round.last().unwrap(), per_round)
}

#[test]
fn host_backward_reduces_loss_sparse_loco() {
    let (first, last, per_round) = loss_decreases_for("sparse_loco");
    assert!(
        first.is_finite() && last.is_finite(),
        "losses are finite (first={first}, last={last})"
    );
    assert!(
        last < first * 0.9,
        "HOST-9: loss must decrease materially (first={first}, last={last}, per_round={per_round:?})"
    );
}

#[test]
fn host_backward_reduces_loss_all_profiles() {
    for profile in ["sparse_loco", "diloco", "demo"] {
        let (first, last, per_round) = loss_decreases_for(profile);
        assert!(
            last < first,
            "HOST-9 [{profile}]: loss must decrease (first={first}, last={last}, per_round={per_round:?})"
        );
    }
}
