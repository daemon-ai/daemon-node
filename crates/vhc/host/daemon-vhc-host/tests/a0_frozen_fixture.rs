// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **The A0 frozen v1 compatibility fixture — expectation FLIPPED at the Phase-E v1 sunset**
//! (decisions D5; refactor §9/§12.2). Historical compatibility tests are never deleted: this is
//! the SAME content-addressed bundle A0 froze (`tests/fixtures/a0-frozen-v1/` — the immutable
//! pre-refactor `tiny-llama` wasm bytes, the exact schema-major-1 signed envelope, the pinned
//! corpus, the recorded transcript), kept as the **standing regression that v1 support is gone
//! and gone gracefully**:
//!
//! - through the sunset the bundle replayed bit-for-bit under the v1 five-phase driver (the
//!   "admitted and green" expectation, decisions D3 cell 1);
//! - since the sunset removed that driver (with `batch_tokens@1`'s driver surface, the autotune
//!   admission, and the phase-legality table, in one auditable step), the SAME pinned module now
//!   meets a **clean, typed `AbiUnsupportedMajor` admission refusal** (ABI §1.5) at the §1.3
//!   front door — a typed `AssessRun` outcome, never a crash, never a silent hang, and no
//!   `da_init`/`da_run`/`da_step` guest code executes.
//!
//! Every content-address pin of the bundle is still verified on every run (the bundle's
//! integrity itself remains regression-tested); only the outcome expectation flipped. The
//! recorded transcript stays in `expected.json` as the permanent historical record of what the
//! v1 driver produced. See the bundle's `README.md` for contents, hashes, and the (pre-sunset)
//! capture command.

use daemon_vhc_abi::AbiRefusalCode;
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::{blake3_hash, from_canonical_slice, SignedEnvelope};

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

/// The flipped expectation (decisions D5): the pinned pre-refactor v1 module is refused with a
/// clean, typed `AbiUnsupportedMajor` — the sunset's permanent regression.
#[test]
fn a0_frozen_fixture_refused_abi_unsupported_major_post_sunset() {
    let expected: serde_json::Value = serde_json::from_str(EXPECTED).expect("expected.json");

    // -- content addressing: every input of the historical bundle is still verified -----------
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
    // The recorded v1 transcript remains the bundle's historical record (never deleted).
    let transcript = expected["transcript"].as_array().unwrap();
    assert_eq!(
        transcript.len() as u64,
        expected["run"]["rounds"].as_u64().unwrap()
    );

    // -- the exact schema-major-1 envelope still opens + signature-verifies --------------------
    // (Envelope v1 support did not retire — a v2 module under a v1 envelope is mixed-fleet
    // cell 5. What retired is the major-1 DRIVER.)
    let wire: SignedEnvelope = from_canonical_slice(ENVELOPE_WIRE).expect("decode SignedEnvelope");
    let frozen = wire.open().expect("open + verify the frozen envelope");
    assert_eq!(
        frozen.hash().to_hex(),
        expected["envelope"]["envelope_hash"].as_str().unwrap()
    );
    assert_eq!(
        blake3_hash(frozen.config_bytes()).to_hex(),
        expected["envelope"]["config_blake3"].as_str().unwrap(),
        "the config byte chain must be intact"
    );
    let envelope = frozen.decode().expect("decode envelope");
    assert_eq!(envelope.run.schema, 1, "the fixture is schema-major 1");
    let module_pin = envelope.artifacts["tiny-llama"].blake3;

    // -- THE FLIP: the §1.3 front door refuses the v1 module with the typed code ---------------
    // The refusal is an admission outcome raised at step 5 (the da_abi cross-check passes —
    // candidate major 1 == declared major 1 — and then the host, which no longer implements
    // major 1, refuses): typed, attributable, pre-guest-execution. Not a trap, not a crash,
    // not BadModule/AbiDeclarationMismatch — exactly `AbiUnsupportedMajor` (ABI §1.5).
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let refusal = select_driver(&worker, MODULE, Some(&module_pin.0))
        .expect_err("a v1 module must be refused on a post-sunset host");
    assert_eq!(
        refusal.code,
        AbiRefusalCode::AbiUnsupportedMajor,
        "the sunset's refusal is the clean typed AbiUnsupportedMajor, got: {refusal}"
    );
    assert!(
        refusal.detail.contains("major 1"),
        "the refusal names the offending declared major (observed vs supported): {}",
        refusal.detail
    );
    assert!(
        refusal.detail.contains("[2]"),
        "the refusal names the host's implemented majors: {}",
        refusal.detail
    );

    // A wrong-pin module is still refused EARLIER (step 1, before any byte reaches the
    // compiler) — the front door's order survives the sunset.
    let refusal = select_driver(&worker, MODULE, Some(&[0u8; 32]))
        .expect_err("a mismatched pin is refused before compile");
    assert_eq!(refusal.code, AbiRefusalCode::ModuleHashMismatch);
}
