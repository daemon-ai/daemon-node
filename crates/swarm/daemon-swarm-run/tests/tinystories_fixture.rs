// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! RUN-3 (TDD §3.3) re-run against the **real vendored TinyStories fixture** — a genuine
//! pre-tokenized corpus (GPT-2 BPE, u16 shards) produced offline by `xtask tokenize-corpus`, so CI
//! never needs egress (spec §8). Provenance (dataset/tokenizer + pinned HF commits) is recorded in
//! the fixture's `manifest.json` and `swarm-ledger-m1.md`.
//!
//! Generation command (documented, reproducible):
//! ```text
//! cargo run -p xtask -- tokenize-corpus \
//!   --dataset roneneldan/TinyStories --dataset-file TinyStories-valid.txt \
//!   --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
//!   --tokenizer gpt2 --tokenizer-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
//!   --out-dir crates/swarm/daemon-swarm-run/tests/fixtures/tinystories \
//!   --shard-tokens 262144 --seq-len 1024 --token-width u16 --max-tokens 1048576
//! ```

use daemon_swarm_run::data::{slice_interval, BatchInterval, BatchLocation, Corpus, Manifest};

const MANIFEST: &str = include_str!("fixtures/tinystories/manifest.json");
const SHARD0: &[u8] = include_bytes!("fixtures/tinystories/shard-0000.bin");
const SHARD1: &[u8] = include_bytes!("fixtures/tinystories/shard-0001.bin");
const SHARD2: &[u8] = include_bytes!("fixtures/tinystories/shard-0002.bin");
const SHARD3: &[u8] = include_bytes!("fixtures/tinystories/shard-0003.bin");

fn manifest() -> Manifest {
    Manifest::from_json(MANIFEST).expect("vendored TinyStories manifest parses + validates")
}

/// The fixture carries the additive provenance so a run is auditable/reproducible.
#[test]
fn fixture_records_pinned_provenance() {
    let m = manifest();
    assert_eq!(m.tokenizer.as_deref(), Some("gpt2"));
    assert_eq!(m.dataset.as_deref(), Some("roneneldan/TinyStories"));
    assert_eq!(
        m.dataset_revision.as_deref(),
        Some("f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"),
        "dataset revision is a pinned HF commit SHA"
    );
    assert!(m.tokenizer_revision.is_some(), "tokenizer revision pinned");
}

/// RUN-3 `manifest_batchid_maps` on the real manifest: `BatchId → (shard, offset)` across shard
/// boundaries, and the out-of-range guard.
#[test]
fn manifest_batchid_maps() {
    let m = manifest();
    // 4 shards × 262144 tokens / seq_len 1024 = 256 sequences each = 1024 total.
    assert_eq!(m.seq_len, 1024);
    assert_eq!(m.shards.len(), 4);
    assert_eq!(m.total_tokens(), 1_048_576);
    assert_eq!(m.total_sequences(), 1024);

    let seq = u64::from(m.seq_len);
    assert_eq!(
        m.locate(0).unwrap(),
        BatchLocation {
            shard: 0,
            token_offset: 0
        }
    );
    // Last sequence of shard 0.
    assert_eq!(
        m.locate(255).unwrap(),
        BatchLocation {
            shard: 0,
            token_offset: 255 * seq
        }
    );
    // First sequence of shard 1 (crossing the shard boundary).
    assert_eq!(
        m.locate(256).unwrap(),
        BatchLocation {
            shard: 1,
            token_offset: 0
        }
    );
    // Somewhere in the last shard.
    assert_eq!(
        m.locate(1000).unwrap(),
        BatchLocation {
            shard: 3,
            token_offset: (1000 - 768) * seq
        }
    );
    // Out of range.
    assert!(m.locate(1024).is_err());
}

/// RUN-3 `interval_slices_into_h_steps` on a real interval at the preset cadence `H = 30`.
#[test]
fn interval_slices_into_h_steps() {
    // 300 sequences over the fixture, 30 inner steps (the sparse_loco preset H), micro-batch 5 →
    // each step is 10 sequences = 2 micro-batches of 5.
    let steps = slice_interval(BatchInterval::new(0, 300), 30, 5).unwrap();
    assert_eq!(steps.len(), 30);
    for (h, step) in steps.iter().enumerate() {
        assert_eq!(step.index, h as u32);
        assert_eq!(step.micro_batches.len(), 2);
        let base = h as u64 * 10;
        assert_eq!(step.micro_batches[0].start, base);
        assert_eq!(step.micro_batches[1].end, base + 10);
    }
    // A non-divisible interval is rejected (cadence is uniform across peers, §6.3).
    assert!(slice_interval(BatchInterval::new(0, 301), 30, 5).is_err());
}

/// The real shards load into a `Corpus` (blake3-verified) and decode to in-range GPT-2 token ids.
#[test]
fn corpus_from_real_shards_reads_tokens() {
    let m = manifest();
    let shards = vec![
        SHARD0.to_vec(),
        SHARD1.to_vec(),
        SHARD2.to_vec(),
        SHARD3.to_vec(),
    ];
    // from_parts verifies each shard's blake3 against the manifest (§8 integrity).
    let corpus = Corpus::from_parts(m, shards).expect("real shards match the manifest blake3");
    assert_eq!(corpus.total_sequences(), 1024);

    let s0 = corpus.sequence(0).unwrap();
    assert_eq!(s0.len(), 1024, "one sequence = seq_len tokens");
    // GPT-2 BPE vocab is 50257 ⇒ every id is a valid u16 embedding index for the 160M preset.
    assert!(s0.iter().all(|&t| t < 50257), "ids within GPT-2 vocab");
    // Wrapping: sequence(total) == sequence(0); a later sequence differs (real text, not constant).
    assert_eq!(corpus.sequence(corpus.total_sequences()).unwrap(), s0);
    assert_ne!(corpus.sequence(600).unwrap(), s0);
}

/// A tampered shard is rejected by the blake3 integrity check (no silent corruption → NaN).
#[test]
fn tampered_shard_rejected() {
    let m = manifest();
    let mut bad = SHARD0.to_vec();
    bad[0] ^= 0xFF;
    let shards = vec![bad, SHARD1.to_vec(), SHARD2.to_vec(), SHARD3.to_vec()];
    assert!(Corpus::from_parts(m, shards).is_err());
}
