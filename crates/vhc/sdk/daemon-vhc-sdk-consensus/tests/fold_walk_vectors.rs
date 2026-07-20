// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The shared **slice-decomposition schedule vectors**
//! (`tests/fixtures/fold-walk-vectors.json`) executed against the normative walk
//! ([`daemon_vhc_sdk_consensus::fold_walk`]) — the pinned contract the multi-slice fold engine
//! and the streaming trainer guest build against.
//!
//! The vectors pin two things:
//!
//! 1. **Window enumeration**: per-parameter chunking in registration order (ordinal ≡ family
//!    chunk ordinal, tails short, a parameter never spans a window) — so the decomposed walk's
//!    operation order is the resident per-parameter order, window-sliced.
//! 2. **The completion-driven schedule**: for an explicit arrival permutation, the exact
//!    per-slice actions — folds ascending and contiguous regardless of arrival order, issues
//!    bounded by `in_flight`, the seal exactly once in the slice folding the last window.
//!
//! A property suite on top proves the fold order is the ascending window order for *arbitrary*
//! arrival permutations and `in_flight` bounds — the invariance that makes the streamed fold
//! deterministic (journal-identical) whatever the read latencies do.

use daemon_vhc_sdk_consensus::fold_walk::{windows, FoldWalk};
use proptest::prelude::*;
use serde_json::Value;

const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/fold-walk-vectors.json"
));

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|n| n.as_u64().expect("uint"))
        .collect()
}

#[test]
fn the_walk_reproduces_every_shared_schedule_vector() {
    let fixture: Value = serde_json::from_str(VECTORS).expect("fixture parses");
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let numels = u64s(&case["numels"]);
        let window_size = case["window_size"].as_u64().expect("window_size");
        let in_flight = case["in_flight"].as_u64().expect("in_flight");

        // 1. The window enumeration is pinned exactly.
        let got = windows(&numels, window_size);
        let want: Vec<Vec<u64>> = case["windows"]
            .as_array()
            .expect("windows")
            .iter()
            .map(u64s)
            .collect();
        assert_eq!(got.len(), want.len(), "vector `{name}`: window count");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(
                [g.ordinal, u64::from(g.param), g.param_off, g.len],
                [w[0], w[1], w[2], w[3]],
                "vector `{name}`: window {}",
                g.ordinal
            );
        }

        // 2. The per-slice schedule under the pinned arrival permutation.
        let arrivals = u64s(&case["arrivals"]);
        let slices = case["slices"].as_array().expect("slices");
        assert_eq!(
            slices.len(),
            arrivals.len() + 1,
            "vector `{name}`: slice count"
        );

        let mut walk = FoldWalk::new(got.len() as u64, in_flight);
        let opening = walk.start();
        let mut actual = vec![opening];
        for &arrival in &arrivals {
            actual.push(
                walk.on_completion(arrival)
                    .unwrap_or_else(|e| panic!("vector `{name}`: completion {arrival}: {e}")),
            );
        }
        for (i, (got, want)) in actual.iter().zip(slices).enumerate() {
            assert_eq!(
                got.fold,
                u64s(&want["fold"]),
                "vector `{name}` slice {i}: fold"
            );
            assert_eq!(
                got.issue,
                u64s(&want["issue"]),
                "vector `{name}` slice {i}: issue"
            );
            assert_eq!(
                got.seal,
                want["seal"].as_bool().expect("seal"),
                "vector `{name}` slice {i}: seal"
            );
        }
        assert!(walk.is_sealed(), "vector `{name}`: walk sealed");
    }
}

proptest! {
    /// For ARBITRARY arrival permutations and in-flight bounds, the concatenated fold order is
    /// exactly the ascending window order, issues never exceed the bound, and the seal fires
    /// exactly once — with the final fold.
    #[test]
    fn fold_order_is_invariant_under_arrival_permutation(
        total in 0u64..24,
        in_flight in 0u64..8,
        seed in 0u64..u64::MAX,
    ) {
        let mut walk = FoldWalk::new(total, in_flight);
        let bound = in_flight.max(1);
        let opening = walk.start();
        prop_assert_eq!(opening.seal, total == 0, "empty walks seal at start");

        let mut outstanding: Vec<u64> = opening.issue.clone();
        prop_assert!(outstanding.len() as u64 <= bound);
        let mut folds: Vec<u64> = Vec::new();
        let mut seals = usize::from(opening.seal);

        // A deterministic pseudo-random arrival order over whatever is outstanding.
        let mut s = seed;
        while !outstanding.is_empty() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let pick = (s >> 33) as usize % outstanding.len();
            let arrival = outstanding.remove(pick);
            let actions = walk.on_completion(arrival).expect("issued completions are accepted");
            folds.extend(&actions.fold);
            outstanding.extend(&actions.issue);
            prop_assert!(outstanding.len() as u64 <= bound, "reads run ahead bounded");
            seals += usize::from(actions.seal);
        }

        prop_assert_eq!(folds, (0..total).collect::<Vec<u64>>(), "ascending fold order");
        prop_assert_eq!(seals, 1, "the seal fires exactly once");
        prop_assert!(walk.is_sealed());
    }
}
