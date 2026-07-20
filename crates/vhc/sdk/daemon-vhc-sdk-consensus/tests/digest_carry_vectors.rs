// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The shared **digest-carry equivalence vectors**
//! (`tests/fixtures/digest-carry-vectors.json`): the proof obligation that the streaming carry
//! over chunk boundaries reproduces the full-coverage round state digest
//! `digest_state(seed, 64, u32::MAX, state)` **bit-for-bit**.
//!
//! Every case pins the digest of a deterministically generated state image (the fixture's
//! seeded splitmix64 f32 generator, per-parameter streams) and the runner proves four
//! independent feeding disciplines reproduce it:
//!
//! 1. the resident oracle (`digest_state` over the flat image — the pinned hex),
//! 2. the carry fed one-shot,
//! 3. the carry fed per fold window (the per-parameter chunking of the det-state contract —
//!    parameter tails, multi-chunk parameters, windows straddling the 64-byte block-index
//!    framing),
//! 4. the carry fed in pathological splits (per-byte, and a fixed 37-byte stride that never
//!    aligns with blocks or windows).
//!
//! A conforming streaming fold implementation must reproduce these values exactly; a mismatch
//! is a fold-engine bug, never a re-pin.

use daemon_vhc_proto::bytes::Seed;
use daemon_vhc_sdk_consensus::{digest_state, DigestCarry};
use serde_json::Value;

const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/digest-carry-vectors.json"
));

/// The digest granularity every case pins (the production digest's block size).
const BLOCK_SIZE: u32 = 64;

/// The fixture's deterministic per-parameter f32-le generator: parameter `i` of a case seeded
/// `case_seed + i + 1` draws `numel` values from a splitmix64-style LCG
/// (`s ← s·6364136223846793005 + 1442695040888963407`; value = `(s >> 40) as f32 / 2^24`).
fn param_bytes(seed: u64, numel: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(usize::try_from(numel * 4).expect("fixture-sized"));
    let mut s = seed;
    for _ in 0..numel {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        #[allow(clippy::cast_precision_loss)]
        out.extend_from_slice(&(((s >> 40) as f32) / 16_777_216.0).to_le_bytes());
    }
    out
}

fn u64_of(v: &Value, key: &str) -> u64 {
    v.get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("vector field `{key}`"))
}

fn numels_of(v: &Value) -> Vec<u64> {
    v["numels"]
        .as_array()
        .expect("numels")
        .iter()
        .map(|n| n.as_u64().expect("numel"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The per-parameter fold windows of a case (the det-state chunking rule: a parameter never
/// spans a window; tails short).
fn fold_windows(params: &[Vec<u8>], window_size: u64) -> Vec<Vec<u8>> {
    let step = usize::try_from(window_size).expect("fixture-sized");
    params
        .iter()
        .flat_map(|p| p.chunks(step))
        .map(<[u8]>::to_vec)
        .collect()
}

#[test]
fn the_carry_reproduces_every_pinned_digest_under_every_split() {
    let fixture: Value = serde_json::from_str(VECTORS).expect("fixture parses");
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let seed = Seed([u8::try_from(u64_of(case, "seed_fill")).expect("fill byte"); 32]);
        let case_seed = u64_of(case, "generator_seed");
        let numels = numels_of(case);
        let window_size = u64_of(case, "window_size");
        let want_hex = case["digest"].as_str().expect("digest");

        let params: Vec<Vec<u8>> = numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_bytes(case_seed + i as u64 + 1, n))
            .collect();
        let flat: Vec<u8> = params.concat();

        // 1. The resident oracle IS the pinned value.
        let oracle = digest_state(&seed, BLOCK_SIZE, u32::MAX, &flat);
        assert_eq!(
            hex(&oracle.0),
            want_hex,
            "vector `{name}`: the resident digest_state oracle must match the pin"
        );

        // 2. One-shot carry.
        let mut carry = DigestCarry::new(&seed, BLOCK_SIZE);
        carry.update(&flat);
        assert_eq!(carry.finalize(), oracle, "vector `{name}`: one-shot carry");
        assert_eq!(carry.bytes_folded(), flat.len() as u64);

        // 3. Per fold window (the streaming discipline: per-parameter chunking, short tails,
        //    windows straddling the 64-byte block framing).
        let mut carry = DigestCarry::new(&seed, BLOCK_SIZE);
        for window in fold_windows(&params, window_size) {
            carry.update(&window);
        }
        assert_eq!(
            carry.finalize(),
            oracle,
            "vector `{name}`: per-window carry (window_size {window_size})"
        );

        // 4. Pathological splits: per-byte, and a 37-byte stride aligned with nothing.
        let mut per_byte = DigestCarry::new(&seed, BLOCK_SIZE);
        for b in &flat {
            per_byte.update(core::slice::from_ref(b));
        }
        assert_eq!(
            per_byte.finalize(),
            oracle,
            "vector `{name}`: per-byte carry"
        );
        let mut stride = DigestCarry::new(&seed, BLOCK_SIZE);
        for chunk in flat.chunks(37) {
            stride.update(chunk);
        }
        assert_eq!(
            stride.finalize(),
            oracle,
            "vector `{name}`: 37-byte-stride carry"
        );
    }
}
