// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The corpus-windowing equivalence oracle** (B2, refactor §6 "corpus windowing, by layer"):
// `daemon-vhc-sdk-v2::corpus` (the SDK/module-policy home the data@ world consumes) must window
// the corpus **decision-for-decision identically** to `daemon-vhc-session::data` (the v1 host
// pipeline, which stays untouched for the retained v1 driver until the Phase-E sunset).
//
// Same pattern as the A2 bridging oracle (relocated round logic ≡ v1 engine): the move is a
// RELOCATION, and this oracle is what makes that claim falsifiable — manifest parsing/validation,
// BatchId location, shard-window coverage (the prefetch request), and token extraction are each
// asserted equal over the same inputs, including the wrap-around and windowed-residency edges.
//
// This leaf test crate is the one place host-side (session) and SDK-side code may link together
// (tracked exception, e2e Cargo.toml). Tier-1: runs in `vhc-ci-det`'s e2e suite (no iroh/live).

use std::collections::BTreeMap;

use daemon_vhc_sdk_v2::corpus::{CorpusWindow as SdkWindow, Manifest as SdkManifest};
use daemon_vhc_session::data::{Corpus as HostCorpus, Manifest as HostManifest, SyntheticCorpus};

/// One corpus both layers read: the session's own synthetic generator (the v1 fetch path's CI
/// stand-in) is the source of truth; the SDK side parses the SAME manifest.json bytes.
fn corpus_fixture(
    seed: u64,
    shards: u32,
    tokens_per_shard: u64,
    seq_len: u32,
) -> (HostManifest, SdkManifest, Vec<(String, Vec<u8>)>) {
    let (host_manifest, blobs) =
        SyntheticCorpus::generate(seed, shards, tokens_per_shard, seq_len).expect("synthetic");
    let json = host_manifest.to_json().expect("manifest json");
    let sdk_manifest = SdkManifest::from_json(&json).expect("sdk parses the v1 manifest.json");
    (host_manifest, sdk_manifest, blobs)
}

/// Manifest arithmetic: totals, per-BatchId location, and out-of-range refusal agree.
#[test]
fn locate_and_totals_agree_across_layers() {
    let (host, sdk, _) = corpus_fixture(0xDAE0_7E57, 3, 45, 9);
    assert_eq!(host.total_sequences(), sdk.total_sequences());
    assert_eq!(host.total_tokens(), sdk.total_tokens());
    for batch in 0..host.total_sequences() {
        let h = host.locate(batch).expect("host locate");
        let s = sdk.locate(batch).expect("sdk locate");
        assert_eq!(
            (h.shard, h.token_offset),
            (s.shard, s.token_offset),
            "locate({batch}) diverged between the host pipeline and the SDK policy"
        );
    }
    // Both refuse one past the end (typed on each side).
    assert!(host.locate(host.total_sequences()).is_err());
    assert!(sdk.locate(sdk.total_sequences()).is_err());
}

/// Shard-window coverage (the module's prefetch REQUEST — architecture §3.2: windowing is
/// policy even when the fetch is mechanism): identical over a sweep including wrap-around.
#[test]
fn shards_covering_agree_across_layers_including_wrap() {
    let (host, sdk, _) = corpus_fixture(0xC0FFEE, 4, 36, 9);
    let total = host.total_sequences();
    for start in 0..total {
        for count in 0..=total + 2 {
            assert_eq!(
                host.shards_covering(start, count),
                sdk.shards_covering(start, count),
                "shards_covering({start}, {count}) diverged"
            );
        }
    }
}

/// Token extraction over a windowed residency: the SDK window reads the same u32 sequences the
/// host's windowed corpus reads (including the wrap), and refuses non-resident shards typed.
#[test]
fn windowed_sequences_agree_across_layers() {
    let (host_manifest, sdk_manifest, blobs) = corpus_fixture(0xD1CE, 4, 36, 9);
    // Stage a strict subset: shards 0 and 2 (a real window shape).
    let mut resident = BTreeMap::new();
    resident.insert(0usize, blobs[0].1.clone());
    resident.insert(2usize, blobs[2].1.clone());
    let host = HostCorpus::windowed(host_manifest, resident.clone()).expect("host windowed");
    let sdk = SdkWindow::assemble(sdk_manifest, resident).expect("sdk window");
    assert_eq!(host.resident_shards(), sdk.resident_shards());

    let per_shard_seqs = 36 / 9;
    for batch in 0..host.total_sequences() * 2 {
        // Which shard the (wrapped) batch addresses decides resident vs typed-refusal.
        let shard = ((batch % host.total_sequences()) / per_shard_seqs) as usize;
        if shard == 0 || shard == 2 {
            assert_eq!(
                host.sequence(batch).expect("host sequence"),
                sdk.sequence(batch).expect("sdk sequence"),
                "sequence({batch}) diverged"
            );
        } else {
            assert!(host.sequence(batch).is_err(), "host resident gate");
            assert!(sdk.sequence(batch).is_err(), "sdk resident gate");
        }
    }
}

/// Integrity at assembly: both layers reject a tampered shard (fetch-time rule, both homes).
#[test]
fn both_layers_reject_tampered_shards() {
    let (host_manifest, sdk_manifest, blobs) = corpus_fixture(0xBAD, 2, 18, 9);
    let mut tampered = BTreeMap::new();
    tampered.insert(0usize, vec![0xFFu8; blobs[0].1.len()]);
    assert!(HostCorpus::windowed(host_manifest, tampered.clone()).is_err());
    assert!(SdkWindow::assemble(sdk_manifest, tampered).is_err());
}
