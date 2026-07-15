// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The A2 choreography-move CROSS-IMPLEMENTATION oracle (refactor §5 A2 item 3): the round logic
// relocated into `daemon-vhc-sdk-rounds` must agree with the v1 engine's, window for window and
// item for item — this crate is the one place allowed to link both sides (its SDK edge is a
// tracked dependency-direction exception), so the comparison lives here:
//
// - assignment + slicing: session::{assignment::interval_for, slice_interval} ≡
//   sdk_rounds::{interval_for, slice_interval} over a seed/roster/window sweep;
// - staging semantics: the engine's record-ordered `StagedPayload` sequence ≡ `Staged::mint`'s
//   record-listed order over the same entries (the Committed<T> re-typing contract's baseline).

use std::collections::BTreeMap;

use daemon_vhc_proto::messages::{BatchWindow, RecordEntry};
use daemon_vhc_proto::{blake3_hash, PeerId, Seed};
use daemon_vhc_sdk_rounds::{
    interval_for as sdk_interval_for, slice_interval as sdk_slice, MicroWindow, Staged,
};
use daemon_vhc_session::backend::StagedPayload;
use daemon_vhc_session::data::{slice_interval as v1_slice, BatchInterval};
use daemon_vhc_session::engine::assignment::interval_for as v1_interval_for;

fn peer(b: u8) -> PeerId {
    PeerId([b; 32])
}

#[test]
fn relocated_assignment_and_slicing_match_the_v1_engine() {
    let roster: Vec<PeerId> = (1u8..=4).map(peer).collect();
    for seed_byte in [0u8, 7, 42, 200] {
        let seed = Seed([seed_byte; 32]);
        let window = BatchWindow { start: 0, end: 64 }; // 4 peers × 16, divisible by 2 steps × mb 4
        for me in &roster {
            let v1 = v1_interval_for(window, seed, &roster, me);
            let sdk = sdk_interval_for(window, seed, &roster, me);
            assert_eq!(
                (v1.start, v1.end),
                (sdk.start, sdk.end),
                "assignment interval must be identical (seed {seed_byte}, peer {:?})",
                me.0[0]
            );

            let v1_steps = v1_slice(BatchInterval::new(v1.start, v1.end), 2, 4).expect("v1 slice");
            let sdk_steps = sdk_slice(
                MicroWindow {
                    start: sdk.start,
                    end: sdk.end,
                },
                2,
                4,
            );
            assert_eq!(v1_steps.len(), sdk_steps.len());
            for (a, b) in v1_steps.iter().zip(&sdk_steps) {
                assert_eq!(a.index, b.index);
                let av: Vec<(u64, u64)> =
                    a.micro_batches.iter().map(|m| (m.start, m.end)).collect();
                let bv: Vec<(u64, u64)> = b.micro.iter().map(|m| (m.start, m.end)).collect();
                assert_eq!(av, bv, "micro-window slicing must be identical");
            }
        }
    }
}

#[test]
fn staged_mint_matches_the_engine_record_ordered_staging() {
    // The engine stages `StagedPayload`s by iterating the record entries in listed order
    // (engine::try_ingest); the SDK `Staged::mint` must produce the identical sequence.
    let entries: Vec<RecordEntry> = [3u8, 1, 2]
        .iter()
        .map(|b| {
            let bytes = vec![*b; 8];
            RecordEntry {
                peer: peer(*b),
                hash: blake3_hash(&bytes),
                size: bytes.len() as u64,
            }
        })
        .collect();
    let mut source: BTreeMap<(u64, PeerId), Vec<u8>> = entries
        .iter()
        .map(|e| ((5, e.peer), vec![e.peer.0[0]; 8]))
        .collect();

    // The v1 sequence, exactly as try_ingest builds it.
    let v1: Vec<StagedPayload> = entries
        .iter()
        .map(|e| StagedPayload {
            peer: e.peer,
            hash: e.hash,
            bytes: vec![e.peer.0[0]; 8],
        })
        .collect();

    let staged = Staged::mint(5, &entries, &mut source).expect("mint");
    assert_eq!(staged.items().len(), v1.len());
    for (a, b) in staged.items().iter().zip(&v1) {
        assert_eq!(a.peer, b.peer);
        assert_eq!(a.hash, b.hash);
        assert_eq!(
            a.bytes, b.bytes,
            "byte-identical staging (Committed<T> baseline)"
        );
    }
}
