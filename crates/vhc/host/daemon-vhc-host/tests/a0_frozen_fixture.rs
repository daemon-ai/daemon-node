// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **The A0 frozen v1 compatibility fixture replay** (refactor §5 A0; decisions D3 cell 1's
//! tier-1 positive pin; invariant 2 "the A0 fixture stays green under the v1 driver").
//!
//! Reloads the content-addressed bundle under `tests/fixtures/a0-frozen-v1/` — the immutable
//! pre-refactor `tiny-llama` wasm bytes (pinned by blake3, captured from the pre-Phase-0-rename
//! tree, never recompiled), the exact schema-major-1 signed envelope, the pinned corpus
//! shard/window derivation, and the recorded expected transcript — and replays it through the
//! **current tree's v1 driver** on the deterministic CPU backend. The per-round payload blake3s
//! and post-ingest det-lane state digests must reproduce **bit-for-bit**. See the bundle's
//! `README.md` for contents, hashes, and the documented capture command.
//!
//! Every later phase must keep this test green until the Phase E v1 sunset flips its expectation
//! to a clean `AbiUnsupportedMajor` refusal (decisions D5 — the fixture is never deleted).

use daemon_vhc_abi::CandidateDriver;
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::{blake3_hash, from_canonical_slice, PeerId, SignedEnvelope};
use daemon_vhc_session::backend::{BatchRef, StagedPayload, StepCtx, TrainerBackend};
use daemon_vhc_session::{WasmBackend, WasmBackendConfig};

const MODULE: &[u8] = include_bytes!("fixtures/a0-frozen-v1/tiny_llama.pre-refactor.wasm");
const ENVELOPE_WIRE: &[u8] = include_bytes!("fixtures/a0-frozen-v1/envelope.signed.cbor");
const EXPECTED: &str = include_str!("fixtures/a0-frozen-v1/expected.json");

// The pinned corpus input — vendored once in this repo (the session tinystories fixture, byte-
// identical to the pre-refactor tree's copy); re-verified against the bundle's recorded hashes
// below, so the reference stays content-addressed.
const CORPUS_MANIFEST: &[u8] =
    include_bytes!("../../daemon-vhc-session/tests/fixtures/tinystories/manifest.json");
const SHARD0: &[u8] =
    include_bytes!("../../daemon-vhc-session/tests/fixtures/tinystories/shard-0000.bin");

/// The pinned batch-derivation rule (documented in `expected.json`/`README.md`; textually
/// mirrors the capture crate): token `i` of batch `b` is
/// `raw[(window_start + b*16 + i) % raw.len()] % vocab`.
fn batch_tokens(
    raw: &[u32],
    window_start: usize,
    vocab: u32,
    seqs: u32,
    seq: u32,
    b: u64,
) -> Vec<u32> {
    let n = (seqs * seq) as usize;
    let base = window_start + (b as usize) * n;
    (0..n)
        .map(|i| raw[(base + i) % raw.len()] % vocab)
        .collect()
}

fn decode_u16_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(2)
        .map(|p| u32::from(u16::from_le_bytes([p[0], p[1]])))
        .collect()
}

#[test]
fn a0_frozen_fixture_replays_v1_driver() {
    let expected: serde_json::Value = serde_json::from_str(EXPECTED).expect("expected.json");

    // -- content addressing: every input is verified against its recorded blake3 -------------
    assert_eq!(
        blake3_hash(MODULE).to_hex(),
        expected["module"]["blake3"].as_str().unwrap(),
        "frozen module bytes must match the recorded pin (immutable pre-refactor bytes)"
    );
    assert_eq!(
        MODULE.len() as u64,
        expected["module"]["bytes"].as_u64().unwrap()
    );
    assert_eq!(
        blake3_hash(ENVELOPE_WIRE).to_hex(),
        expected["envelope"]["wire_blake3"].as_str().unwrap(),
        "envelope wire bytes must match the recorded pin"
    );
    assert_eq!(
        blake3_hash(CORPUS_MANIFEST).to_hex(),
        expected["corpus"]["manifest_blake3"].as_str().unwrap(),
        "corpus manifest must match the recorded pin"
    );
    assert_eq!(
        blake3_hash(SHARD0).to_hex(),
        expected["corpus"]["shard0_blake3"].as_str().unwrap(),
        "corpus shard 0 must match the recorded pin"
    );

    // -- the exact schema-major-1 envelope opens + signature-verifies on the current tree ------
    let wire: SignedEnvelope = from_canonical_slice(ENVELOPE_WIRE).expect("decode SignedEnvelope");
    let frozen = wire.open().expect("open + verify the frozen envelope");
    assert_eq!(
        frozen.hash().to_hex(),
        expected["envelope"]["envelope_hash"].as_str().unwrap()
    );
    assert_eq!(
        blake3_hash(frozen.config_bytes()).to_hex(),
        expected["envelope"]["config_blake3"].as_str().unwrap(),
        "the da_build config byte chain must be intact"
    );
    let envelope = frozen.decode().expect("decode envelope");
    assert_eq!(envelope.run.schema, 1, "the fixture is schema-major 1");
    let module_pin = envelope.artifacts["tiny-llama"].blake3;

    // -- the A0 dual-dispatch front door admits the pinned module through the v1 driver --------
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, MODULE, Some(&module_pin.0))
        .expect("the frozen v1 module must pass ABI §1.3 selection");
    assert_eq!(sel.driver, CandidateDriver::V1);
    assert_eq!((sel.major, sel.minor), (1, 0));

    // -- replay: the recorded run, re-executed on today's v1 driver ----------------------------
    let run = &expected["run"];
    let corpus = &expected["corpus"];
    let rounds = run["rounds"].as_u64().unwrap();
    let seqs = run["seqs_per_batch"].as_u64().unwrap() as u32;
    let seq = run["seq_len"].as_u64().unwrap() as u32;
    let window_start = corpus["window_start"].as_u64().unwrap() as usize;
    let vocab = corpus["vocab_mod"].as_u64().unwrap() as u32;
    let raw = decode_u16_le(SHARD0);
    assert_eq!(raw.len() as u64, corpus["shard0_tokens"].as_u64().unwrap());

    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: MODULE.to_vec(),
        engine: EngineConfig::default(),
    })
    .expect("construct WasmBackend");
    backend.build(frozen.config_bytes()).expect("da_build");
    let steps = backend.steps_per_round().expect("steps_per_round");
    assert_eq!(u64::from(steps), run["steps_per_round"].as_u64().unwrap());

    let transcript = expected["transcript"].as_array().unwrap();
    assert_eq!(transcript.len() as u64, rounds);
    let mut digests = Vec::new();
    for round in 0..rounds {
        for step in 0..steps {
            let b = BatchRef {
                tokens: batch_tokens(
                    &raw,
                    window_start,
                    vocab,
                    seqs,
                    seq,
                    round * u64::from(steps) + u64::from(step),
                ),
                seq_len: seq,
            };
            let ctx = StepCtx {
                inner_step: step,
                mb_index: 0,
                mb_count: 1,
                step_seqs: seqs,
            };
            backend.train_step(&b, ctx).expect("train_step");
            backend.inner_update(step).expect("inner_update");
        }
        let payload = backend.make_update(round).expect("make_update");
        let want = &transcript[round as usize];
        assert_eq!(
            blake3_hash(&payload).to_hex(),
            want["payload_blake3"].as_str().unwrap(),
            "round {round}: the sealed round payload must reproduce the pre-refactor bytes"
        );
        let staged = StagedPayload {
            peer: PeerId([1; 32]),
            hash: blake3_hash(&payload),
            bytes: payload,
        };
        let digest = backend.ingest(round, &[staged]).expect("ingest");
        assert_eq!(
            digest.to_hex(),
            want["digest"].as_str().unwrap(),
            "round {round}: the post-ingest det-lane state digest must reproduce bit-for-bit"
        );
        digests.push(digest);
    }
    // Non-degenerate: the pinned transcript genuinely evolves.
    assert!(
        digests.windows(2).any(|w| w[0] != w[1]),
        "the fixture transcript must evolve across rounds"
    );
}
